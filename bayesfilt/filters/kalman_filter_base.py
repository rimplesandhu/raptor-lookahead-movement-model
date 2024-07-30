""" Kalman filter base class """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from collections.abc import Sequence
from abc import abstractmethod
from copy import deepcopy
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from .utils import get_covariance_ellipse, validate_array, sym_posdef_matrix
from .filter_attributes_static import FilterAttributesStatic
from .filter_attributes_dynamic import FilterAttributesDynamic


class KalmanFilterBase(FilterAttributesStatic, FilterAttributesDynamic):
    """ Base class for implementing various versions of Kalman Filters"""

    def __init__(
        self,
        *args,
        **kwargs
    ):
        FilterAttributesStatic.__init__(self, *args, **kwargs)
        FilterAttributesDynamic.__init__(self)

    # def __repr__(self):
    #     """Updated repr function"""
    #     istr = super().__repr__()
    #     return f'----Kalman Filtering----\n{istr}\n'

    def initiate(
        self,
        t0: float | None = None,
        m0: ndarray | None = None,
        P0: ndarray | None = None
    ) -> None:
        """Initiate the filter"""
        if t0 is None:
            if self.time_elapsed is None:
                self.raiseit('Need to initiate time')
        else:
            self._start_time = t0
            self._time_elapsed = t0
            self._last_update_at = t0
        if m0 is None:
            if self.m is None:
                self.raiseit('Need to initiate state mean m')
        else:
            self.m = m0
        if P0 is None:
            if self.P is None:
                self.raiseit('Need to initiate state cov P')
        else:
            self.P = P0
        self._dfraw.clear()
        self.cur_metrics = self.compute_metrics()
        self.store_this_timestep()

    def initiate_state(
        self,
        dict_of_mean_std: dict[str, tuple[float, float]]
    ) -> None:
        """Initiate the time, state mean and covariance matrix"""
        self.m = np.zeros((self.nx,))
        self.P = np.eye(self.nx)
        for i, iname in enumerate(self.state_names):
            if iname in dict_of_mean_std.keys():
                self.m[i] = dict_of_mean_std[iname][0]
                self.P[i, i] = dict_of_mean_std[iname][1]**2

    @abstractmethod
    def forecast(self) -> None:
        """Forecast step"""
        self._time_elapsed += self.dt
        self._y = None
        self.cur_metrics = self.compute_metrics()

    @abstractmethod
    def update(self) -> None:
        """Update step"""

    @abstractmethod
    def _backward_filter(self, smean_next, scov_next) -> None:
        """Backward filter"""

    def forecast_upto(
        self,
        upto_time: float
    ) -> None:
        """Kalman filter forecast step upto some time in future"""
        time_diff = upto_time - self.time_elapsed
        if abs(time_diff) > self.dt_tol:
            if time_diff < 0.:
                istr = f'{self.time_elapsed}->{upto_time}'
                self.raiseit(f'Forecasting backward in time:{istr}!')
            nsteps = int(np.round(time_diff / self.dt))
            for _ in range(nsteps):
                self.forecast()

    def filter(
        self,
        list_of_time: Sequence[float],
        list_of_obs: Sequence[ndarray],
        list_of_R: Sequence[ndarray] | None = None,
    ) -> None:
        """Run filtering assuming F, H, Q matrices are time invariant"""

        self.initiate()
        nsteps = len(list_of_time)
        if len(list_of_obs) != nsteps:
            istr = 'Length mismatch between list_of_time and list_of_obs'
            self.raiseit(f'{istr}: {nsteps} vs {len(list_of_obs)}')
        if list_of_R is not None:
            if len(list_of_R) != nsteps:
                istr = 'Length mismatch between list_of_time and list_of_R'
                self.raiseit(f'{istr}: {nsteps} vs {len(list_of_R)}')
        else:
            if self.R is None:
                self.raiseit('Need to initiate obs cov matrix R')

        tloop = enumerate(list_of_time)
        if self.verbose:
            tloop = tqdm(enumerate(list_of_time), total=len(list_of_time),
                         desc=self.__class__.__name__)
        for k, itime in tloop:
            if self.time_elapsed - itime > self.dt_tol:
                istr = f'y at {itime}, x at {self.time_elapsed}'
                self.raiseit(f'Skipping observation, {istr}!')
            if itime - self.time_elapsed >= self.dt:
                self.forecast_upto(itime)
                self.P = sym_posdef_matrix(self.P)
            self.y = list_of_obs[k]
            if list_of_R is not None:
                self.R = list_of_R[k]
            self.update()

    def smoother(self):
        """Run smoothing assuming model/measurement eq are time invariant"""
        nsteps = len(self.dfraw[self.time_colname])
        if nsteps < 2:
            self.raiseit('No state history found, run filter() first!')
        colnames = [self.smean_colname, self.scov_colname,
                    self.smetrics_colname]
        for cname in colnames:
            self._dfraw[cname] = []
        self.cur_smetrics = self.compute_metrics()
        tloop = reversed(range(nsteps))
        if self.verbose:
            tloop = tqdm(reversed(range(nsteps)), total=nsteps,
                         desc=self.__class__.__name__ + 'S')
        for i in tloop:
            self.m = deepcopy(self._dfraw[self.mean_colname][i])
            self.P = deepcopy(self._dfraw[self.cov_colname][i])
            self.y = deepcopy(self._dfraw[self.y_colname][i])
            self.R = deepcopy(self._dfraw[self.ycov_colname][i])
            if i != nsteps - 1:
                smean_next = self._dfraw[self.smean_colname][-1]
                scov_next = self._dfraw[self.scov_colname][-1]
                self._backward_filter(smean_next, scov_next)
            self._dfraw[self.smean_colname].append(deepcopy(self.m))
            self._dfraw[self.scov_colname].append(deepcopy(self.P))
            self._dfraw[self.smetrics_colname].append(self.cur_smetrics)
        for cname in colnames:
            self._dfraw[cname].reverse()

    def compute_metrics(
        self,
        xres: ndarray | None = None,
        xprec: ndarray | None = None,
        yres: ndarray | None = None,
        yprec: ndarray | None = None
    ):
        """compute filter performance metrics"""
        idict = {
            'MetricXresNorm': np.nan,
            'MetricYresNorm': np.nan,
            'MetricNIS': np.nan,
            'MetricNEES': np.nan,
            'MetricLogLik': np.nan
        }
        if (yres is not None) and (yprec is not None):
            idict['MetricYresNorm'] = np.linalg.norm(yres)
            idict['MetricNIS'] = np.linalg.multi_dot([yres.T, yprec, yres])
            prec_det = np.linalg.det(yprec)
            idict['MetricLogLik'] = -0.5 * (self.ny * np.log(2. * np.pi) -
                                            np.log(prec_det) + idict['MetricNIS'])
        if (xres is not None) and (xprec is not None):
            idict['MetricXresNorm'] = np.linalg.norm(xres)
            idict['MetricNEES'] = np.linalg.multi_dot([xres.T, xprec, xres])
        return idict

    def get_loglik_of_obs(
        self,
        y_obs: ndarray,
        ignore_obs_inds: list[int] | None = None
    ) -> None:
        """
        Compute log-likelihood of this observation
        """
        y_pred = self.mat_H @ self.m
        _residual = y_obs - y_pred
        _this_smat = self.mat_H @ self.P @ self.mat_H.T + self.R
        if ignore_obs_inds is not None:
            for idx in ignore_obs_inds:
                _residual[idx] = 0.
        _s_inv = np.linalg.pinv(_this_smat, hermitian=True)
        # _s_inv = np.linalg.inv(_this_smat)
        # print(_s_inv.diagonal())
        this_loglik = self.ny * np.log(2. * np.pi)
        this_loglik += np.log(np.linalg.det(_this_smat))
        this_loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
        this_loglik = np.linalg.multi_dot([_residual.T, _s_inv, _residual])
        # this_loglik = np.dot(_residual, np.dot(_s_inv, _residual))
        # this_loglik = np.dot(_residual, _residual)
        this_loglik *= -0.5
        return this_loglik

    def symmetrize(self, in_mat: ndarray) -> ndarray:
        """Return a symmetrized, regularized covariance matrix"""
        return (in_mat + in_mat.T) / 2. + np.diag([self.epsilon] * self.nx)

    def get_state_index(self, state: int | str) -> int:
        """Get state index and name"""
        if isinstance(state, int):
            if state > self.nx:
                self.raiseit(f'Invalid index {state}, choose < {self.nx}')
            idx = state
        elif isinstance(state, str):
            if state not in self.state_names:
                self.raiseit(f'Invalid {state}, Valid: {self.state_names}')
            idx = self.state_names.index(state)
        return idx

    def get_mean(
        self,
        state: int | str,
        smoother: bool = False
    ) -> ndarray:
        """Get time series of state mean for a given index/name of the state"""
        colname = self.smean_colname if smoother else self.mean_colname
        if colname not in self.dfraw.columns:
            self.raiseit('No smoother history found, run smoother() first!')
        idx = self.get_state_index(state)
        return np.stack(self.dfraw[colname].values)[:, idx]

    def get_cov(
        self,
        state: str | int | tuple[int, int] | tuple[str, str],
        smoother: bool = False
    ) -> ndarray:
        """Get state covariance matrix for the desired indices"""
        cname = self.scov_colname if smoother else self.cov_colname
        if isinstance(state, int) | isinstance(state, str):
            idx = self.get_state_index(state)
            idx_pair = [idx, idx]
        elif isinstance(state, Sequence):
            assert len(state) == 2, 'Need a pair of state idx/names!'
            idx_pair = [0, 0]
            for i, idx in enumerate(state):
                idx_pair[i] = self.get_state_index(idx)
        else:
            self.raiseit('Need either int or str or (int,int) or (str,str)')
        return np.stack(self.dfraw[cname].values)[:, idx_pair[0], idx_pair[1]]

    def get_df(
        self,
        smoother=False,
        variance: bool = False,
        metrics: bool = True
    ) -> pd.DataFrame:
        """Returns short version of pandas dataframe"""
        out_df = self.dfraw.loc[:, [self.time_colname]].copy()
        mcol = self.smetrics_colname if smoother else self.metrics_colname
        for iname in self.state_names:
            out_df[iname] = self.get_mean(iname, smoother=smoother)
            if variance:
                out_df[iname + '_var'] = self.get_cov(iname, smoother=smoother)
        # dff['Observation'] = self.dfraw['Observation']
        out_df['ObjectId'] = self.objectid
        if metrics:
            _mdf = self.dfraw[mcol].apply(pd.Series)
            out_df = pd.concat([out_df, _mdf], axis=1)
        return out_df

    @property
    def df(self) -> pd.DataFrame:
        """History of the filter in the form of a pandas dataframe"""
        return self.get_df(smoother=False)

    @property
    def dfs(self) -> pd.DataFrame:
        """History of the filter in the form of a pandas dataframe"""
        return self.get_df(smoother=True)

    def plot_state_mean(
        self,
        state,
        *args,
        ax: plt.Axes | None = None,
        smoother: bool = False,
        **kwargs
    ) -> None:
        """Plot the time history of state estimates"""
        ax = plt.gca() if ax is None else ax
        xvec = self.get_mean(state, smoother=smoother)
        cb = ax.plot(self.tlist, xvec, *args, **kwargs)
        state_idx = self.get_state_index(state)
        ax.set_ylabel(f'{self.state_names[state_idx]}')
        ax.set_xlim([self.tlist[0], self.tlist[-1]])
        ax.set_xlabel('Time elapsed (Seconds)')
        return cb

    def plot_state_cbound(
        self,
        state,
        *args,
        ax: plt.Axes | None = None,
        smoother: bool = False,
        cb_fac: float = 3.,
        **kwargs
    ) -> None:
        """Plot the time history of state estimates"""
        ax = plt.gca() if ax is None else ax
        xvec = self.get_mean(state, smoother=smoother)
        xvec_var = self.get_cov(state, smoother=smoother)
        cb_width = np.ravel(cb_fac * xvec_var**0.5)
        cb = ax.fill_between(self.tlist, xvec - cb_width, xvec +
                             cb_width, *args, **kwargs)
        state_idx = self.get_state_index(state)
        ax.set_ylabel(f'{self.state_names[state_idx]}')
        ax.set_xlim([self.tlist[0], self.tlist[-1]])
        ax.set_xlabel('Time elapsed (Seconds)')
        return cb

    def plot_trajectory_cbound(
        self,
        x_indx: int,
        y_indx: int,
        ax: plt.Axes | None = None,
        smoother: bool = False,
        cb_fac: float = 3.,
        **kwargs
    ) -> None:
        """Plot the trajectory"""
        ax = plt.gca() if ax is None else ax
        xlocs = self.get_mean(x_indx, smoother=smoother)
        ylocs = self.get_mean(y_indx, smoother=smoother)
        xy_vars = self.get_cov([x_indx, y_indx], smoother=smoother)
        for xloc, yloc, icov in zip(xlocs, ylocs, xy_vars):
            width, height, angle = get_covariance_ellipse(icov, cb_fac)
            ellip = Ellipse(xy=[xloc, yloc], width=width, height=height,
                            angle=angle, **kwargs)
            ax.add_artist(ellip)

    def plot_trajectory_mean(
        self,
        x_indx: int,
        y_indx: int,
        *args,
        ax: plt.Axes | None = None,
        smoother: bool = False,
        **kwargs
    ) -> None:
        """Plot the trajectory"""
        ax = plt.gca() if ax is None else ax
        xlocs = self.get_mean(x_indx, smoother=smoother)
        ylocs = self.get_mean(y_indx, smoother=smoother)
        cb = ax.plot(xlocs, ylocs, *args, **kwargs)
        return cb

    @property
    def metrics(self) -> dict[str, float]:
        """Return the summary metrics"""
        idf = self.df
        metric_names = list(self.cur_metrics.keys())
        return idf.loc[~idf[metric_names[0]].isna(), metric_names].sum()

    @property
    def smetrics(self) -> dict[str, float]:
        """Return the summary metrics"""
        idf = self.dfs
        metric_names = list(self.cur_smetrics.keys())
        return idf.loc[~idf[metric_names[0]].isna(), metric_names].sum()

    def plot_metric(
        self,
        metric: str = 'NIS',
        smoother=False,
        ax: plt.Axes | None = None,
        **kwargs
    ) -> None:
        """Plot the time history of a performance metric"""
        ax = plt.gca() if ax is None else ax
        mnames = list(self.cur_metrics.keys())
        if metric not in mnames:
            self.raiseit(f'Invalid metric {metric}, choose from {mnames}')
        idf = self.dfs if smoother else self.df
        idfshort = idf[~idf[metric].isna()]
        tvec = idfshort[self.time_colname].values
        xvec = idfshort[metric].values
        ax.plot(tvec, xvec, **kwargs)
        ax.set_ylabel(metric)
        ax.set_xlim([tvec[0], tvec[-1]])
        ax.set_xlabel('Time elapsed (Seconds)')

    @property
    def y(self) -> ndarray:
        """Observation vector"""
        return self._y

    @property
    def m(self) -> ndarray:
        """State mean vector"""
        return self._m

    @property
    def P(self) -> ndarray:
        """State covariance matrix"""
        return self._P

    @property
    def R(self) -> ndarray:
        """Observation error covariance matrix"""
        return self._R

    @property
    def get_time_elapsed(self) -> ndarray:
        """Get time elapsed"""
        return self.df[self.time_colname].values

    @property
    def start_time(self) -> ndarray:
        """Get time elapsed"""
        return self._start_time

    @property
    def lifespan_to_last_update(self) -> float:
        """Returns the time duration of existence till the last update"""
        return self.last_update_at - self.start_time

    @property
    def lifespan_to_last_forecast(self) -> float:
        """Returns the time duration of existence till the last update"""
        return self.time_elapsed - self.start_time

    @m.setter
    def m(self, in_vec: ndarray) -> None:
        """Setter for m vector"""
        self._m = validate_array(in_vec, self.nx, return_array=True)

    @P.setter
    def P(self, in_mat: ndarray | None) -> None:
        """Setter for m vector"""
        self._P = validate_array(in_mat, (self.nx, self.nx), return_array=True)

    @R.setter
    def R(self, in_mat: ndarray | None) -> None:
        """Setter for m vector"""
        if in_mat is not None:
            self._R = validate_array(in_mat, (self.ny, self.ny),
                                     return_array=True)
        else:
            self._R = None

    @y.setter
    def y(self, in_vec: ndarray | None) -> None:
        """Setter for m vector"""
        if in_vec is not None:
            self._y = validate_array(in_vec, self.ny, return_array=True)
        else:
            self._y = None


# def initiate_state_by_dict(
#     self,
#     dict_of_mean_std: dict[str, tuple[float, float]]
# ) -> None:
#     """Initiate the time, state mean and covariance matrix"""
#     self._m = np.zeros((self.nx,))
#     self._P = np.eye(self.nx)
#     for i, iname in enumerate(self.state_names):
#         if iname in dict_of_mean_std.keys():
#             self._m[i] = dict_of_mean_std[iname][0]
#             self._P[i, i] = dict_of_mean_std[iname][1]**2
