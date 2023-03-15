""" Kalman filter base class """
from collections.abc import Sequence, Callable
from typing import Dict, Tuple, List
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class KalmanFilterBase(ABC):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    # pylint: disable=invalid-name
    """
    Base class for implementing various Gaussian Kalman Filters
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0
    ):
        # basic info
        self.name: str = 'KFBase'
        self.id: str = object_id
        self.id_col = 'ObjectID'
        self.dt: float = dt
        self.dt_tol: float = 0.01
        self.epsilon: float = 1e-20
        self._nx: int = self.int_setter(nx)  # dimension of state vector
        self._ny: int = self.int_setter(ny)  # dimension of observation vector
        self._m: ndarray | None = None  # state mean vector
        self._P: ndarray | None = None  # state covariance matrix
        self._truth: ndarray | None = None
        self._obs: ndarray | None = None
        self.pars: Dict[str, float] = {}

        # dynamics model
        self.func_f: Callable | None = None  # state transition function
        self._F: ndarray | None = None  # state transition matrix
        self.func_Q: Callable | None = None
        self._Q: ndarray | None = None  # process error cov mat
        self._G: ndarray = np.eye(self.nx)  # Error jacobian matrix
        self._qbar: ndarray = np.zeros((self.nx,))  # process error cov mat

        # observation model
        self.func_h: Callable | None = None  # observation-state function
        self._H: ndarray | None = None  # observation-state matrix
        self.func_R: Callable | None = None
        self._R: ndarray | None = None  # obs error covariance matrix
        self._J: ndarray = np.eye(self.ny)  # obs error jacobian matrix
        self._rbar: ndarray = np.zeros((self.ny,))  # obs error mean

        # add/subtract functions
        self.x_add: Callable = np.add
        self.x_subtract: Callable = np.subtract
        self.y_subtract: Callable = np.subtract
        self.x_mean_fn: Callable | None = None
        self.y_mean_fn: Callable | None = None

        # Kalman filtering parameters
        self._nees: float = np.nan  # normalized estimation error squared
        self._nis: float = np.nan  # normalized innovation squared
        self._loglik: float = np.nan  # log lik of observations
        self._time_elapsed: float = 0.  # time elapsed, default starts at 0.
        self._last_update_at: float = 0.  # last update at this time
        self._state_names: Sequence[str] = [f'x_{i}' for i in range(self._nx)]

        # saving history of filter and smoother
        varnames = ['TimeElapsed', 'Observation', 'Truth', 'ObservationCov',
                    'FilterMean', 'FilterCov']
        self.filter_metrics = ['FilterNEES', 'FilterNIS', 'FilterLogLik']
        self.smoother_metrics = ['SmootherNEES',
                                 'SmootherNIS', 'SmootherLogLik']
        self._history = {}
        for varname in varnames + self.filter_metrics:
            self._history[varname] = []

    def initiate_state(
        self,
        t0: float,
        m0: ndarray,
        P0: ndarray | None = None
    ) -> None:
        """Initiate the time, state mean and covariance matrix"""
        for v in self._history.values():
            v.clear()
        self._time_elapsed = t0
        self._m = self.vec_setter(m0, self.nx)
        if P0 is not None:
            self._P = self.mat_setter(P0, (self.nx, self.nx))
        self._truth = None
        self._obs = None
        self._nees = np.nan
        self._nis = np.nan
        self._loglik = np.nan
        self._last_update_at = t0
        self._store_this_step()

    def initiate_state_by_dict(
        self,
        t0: float,
        dict_of_mean_std: Dict[str, Tuple[float, float]]
    ) -> None:
        """Initiate the time, state mean and covariance matrix"""
        for v in self._history.values():
            v.clear()
        self._time_elapsed = t0
        self._m = np.zeros((self.nx,))
        self._P = np.eye(self.nx)
        for i, iname in enumerate(self.state_names):
            if iname in dict_of_mean_std.keys():
                self._m[i] = dict_of_mean_std[iname][0]
                self._P[i, i] = dict_of_mean_std[iname][1]**2
        self._truth = None
        self._obs = None
        self._nees = np.nan
        self._nis = np.nan
        self._loglik = np.nan
        self._store_this_step()

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
        list_of_truth: Sequence[ndarray] | None = None
    ) -> None:
        """Run filtering assuming F, H, Q matrices are time invariant"""

        # check if inputs are valid
        nsteps = len(list_of_time)
        if len(list_of_obs) != nsteps:
            print('Length mismatch!')
            self.raiseit(f'time:{nsteps} vs observations: {len(list_of_obs)}')
        if list_of_R is not None:
            if len(list_of_R) != nsteps:
                self.raiseit('Length mismatch: time vs list of R matrices')
        if list_of_truth is not None:
            if len(list_of_truth) != nsteps:
                self.raiseit('Size mismatch: time vs list of state truths')
        if len(self._history['TimeElapsed']) > 1:
            self.raiseit('Need to initiate state!')

        # run the forward filter
        k = 0
        while k < len(list_of_time):
            if self._time_elapsed - list_of_time[k] > self.dt_tol:
                self.raiseit(f'Skipping observation, lower the dt={self.dt}!')
            if abs(self._time_elapsed - list_of_time[k]) < self.dt_tol:
                self.obs = list_of_obs[k]
                self.R = self.R if list_of_R is None else list_of_R[k]
                self.truth = None if list_of_truth is None else list_of_truth[k]
                self.update()
                k += 1
            else:
                self.forecast()
                if np.any(np.linalg.eigvals(self.P) < 0):
                    print('Exiting because covariance is not pos def!')
                    break

    def smoother(self):
        """Run smoothing assuming model/measurement eq are time invariant"""
        nsteps = len(self.history['TimeElapsed'])
        if nsteps < 1:
            self.raiseit('No state history found, run filter first!')
        cnames = ['SmootherMean', 'SmootherCov'] + self.smoother_metrics
        for cname in cnames:
            self._history[cname] = []
        for i in reversed(range(nsteps)):
            self._time_elapsed = self.history['TimeElapsed'][i]
            self._obs = self.history['Observation'][i]
            self._truth = self.history['Truth'][i]
            self._m = deepcopy(self.history['FilterMean'][i])
            self._P = deepcopy(self.history['FilterCov'][i])
            self._R = self.history['ObservationCov'][i]
            if i != nsteps - 1:
                smean_next = self.history['SmootherMean'][-1]
                scov_next = self.history['SmootherCov'][-1]
                self._backward_filter(smean_next, scov_next)
            self._history['SmootherMean'].append(deepcopy(self.m))
            self._history['SmootherCov'].append(deepcopy(self.P))
            self._history[self.smoother_metrics[0]].append(self.nees)
            self._history[self.smoother_metrics[1]].append(self.nis)
            self._history[self.smoother_metrics[2]].append(self.loglik)
        for k, v in self._history.items():
            if k.startswith('Smoother'):
                v.reverse()

    def _store_this_step(self, update: bool = False) -> None:
        """Store this forecast/update step"""
        self._time_elapsed = np.around(self.time_elapsed, 3)
        if update:
            for k, v in self._history.items():
                if not k.startswith('Smoother'):
                    del v[-1]
            self._last_update_at = self.time_elapsed
        self._history['TimeElapsed'].append(self.time_elapsed)
        self._history['Observation'].append(self.obs)
        self._history['Truth'].append(self.truth)
        self._history['ObservationCov'].append(self.R)
        self._history['FilterMean'].append(deepcopy(self.m))
        self._history['FilterCov'].append(deepcopy(self.P))
        self._history[self.filter_metrics[0]].append(self.nees)
        self._history[self.filter_metrics[1]].append(self.nis)
        self._history[self.filter_metrics[2]].append(self.loglik)

    def _compute_metrics(self, xres, xprec, yres, yprec):
        """compute performance metrics"""
        self._nis = np.linalg.multi_dot([yres.T, yprec, yres])
        prec_det = np.linalg.det(yprec)
        # if prec_det > 0.:
        self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) -
                               np.log(prec_det) + self.nis)
        # else:
        #    self._loglik = np.nan
        if self.truth is not None:
            xres = self.m - self.truth
        self._nees = np.linalg.multi_dot([xres.T, xprec, xres])

    def get_loglik_of_obs(
        self,
        y_obs: ndarray,
        ignore_obs_inds: List[int] | None = None
    ) -> None:
        """
        Compute log-likelihood of this observation
        """
        y_pred = self.H @ self.m
        _residual = y_obs - y_pred
        _this_smat = self.H @ self.P @ self.H.T + self.R
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

### abstract methods to be implemented by children of this base class ###

    @ abstractmethod
    def validate(self) -> None:
        """Check if relevant matrices and/or functions have been initiated"""
        if (self.P is None) or (self.m is None):
            self.raiseit('Need to initiate state, use initiate_state()')
        if self.R is None:
            self.raiseit('Need to initiate R matrix!')
        if self.Q is None:
            self.raiseit('Need to initiate Q matrix!')

    @ abstractmethod
    def forecast(self) -> None:
        """Forecast step"""
        self._time_elapsed += np.around(self.dt, 3)
        self._truth = None
        self._obs = None
        self._nis = np.nan
        self._nees = np.nan
        self._loglik = np.nan

    @ abstractmethod
    def update(self) -> None:
        """Update step"""

    @ abstractmethod
    def _backward_filter(self, smean_next, scov_next) -> None:
        """Backward filter"""


### accessing filter/smoother results###

    def get_mean(
        self,
        this_state: str | int,
        smoother: bool = False
    ) -> ndarray:
        """Get state mean time series of ith state"""
        cname = 'SmootherMean' if smoother else 'FilterMean'
        if isinstance(this_state, int):
            idx = this_state
            self.check_state_index(idx)
        elif isinstance(this_state, str):
            assert this_state in self.state_names, f'{this_state} invalid!'
            idx = self.state_names.index(this_state)
        return np.stack(self.df[cname].values)[:, idx]

    def get_time_elapsed(self) -> ndarray:
        """Get time elapsed"""
        return self.df['TimeElapsed'].values

    def get_cov(
        self,
        this_state: str | int | Tuple[int, int] | Tuple[str, str],
        smoother: bool = False
    ) -> ndarray:
        """Get state covariance matrix for the desired indices"""
        cname = 'SmootherCov' if smoother else 'FilterCov'
        if isinstance(this_state, int):
            idx_pair = [this_state, this_state]
        elif isinstance(this_state, str):
            assert this_state in self.state_names, f'{this_state} invalid!'
            idx = self.state_names.index(this_state)
            idx_pair = [idx, idx]
        elif isinstance(this_state, Sequence):
            assert len(this_state) == 2, 'Need a pair of state idx/names!'
            idx_pair = [0, 0]
            for i, idx in enumerate(this_state):
                if isinstance(idx, int):
                    self.check_state_index(idx)
                    idx_pair[i] = idx
                elif isinstance(idx, str):
                    self.check_state_name(idx)
                    idx_pair[i] = self.state_names.index(idx)
                else:
                    self.raiseit('Need either int or str!')
        else:
            self.raiseit('Need either int or str or (int,int) or (str,str)')
        # indices = np.ix_(np.arange(self.df.shape[0]), idx_pair[0], idx_pair[1])
        return np.stack(self.df[cname].values)[:, idx_pair[0], idx_pair[1]]

    # def get_cov(
    #     self,
    #     row_inds: int,
    #     col_inds: int | None = None,
    #     smoother: bool = False
    # ) -> ndarray:
    #     """Get state covariance matrix for the desired indices"""
    #     cname = 'SmootherCov' if smoother else 'FilterCov'
    #     row_inds = row_inds if isinstance(row_inds, list) else [row_inds]
    #     col_inds = col_inds if col_inds is not None else row_inds
    #     col_inds = col_inds if isinstance(col_inds, list) else [col_inds]
    #     indices = np.ix_(np.arange(self.df.shape[0]), row_inds, col_inds)
    #     return np.stack(self.df[cname].values)[indices]

    @ property
    def metrics(self) -> dict:
        """Get the summary performance metrics"""
        out_metrics = {}
        for cname in self.filter_metrics + self.smoother_metrics:
            if cname in list(self.df.columns):
                out_metrics[cname] = np.around(
                    self.df[cname].dropna().sum(), 3)
        return out_metrics


### Plotting related ###

    def plot_state_mean(
        self,
        this_state,
        *args,
        ax: plt.Axes | None = None,
        smoother: bool = False,
        **kwargs
    ) -> None:
        """Plot the time history of state estimates"""
        ax = plt.gca() if ax is None else ax
        tvec = self.get_time_elapsed()
        xvec = self.get_mean(this_state, smoother=smoother)
        cb = ax.plot(tvec, xvec, *args, **kwargs)
        # ax.set_ylabel(f'{self.state_names[state_idx]}')
        ax.set_xlim([tvec[0], tvec[-1]])
        # ax.set_xlabel('Time elapsed (Seconds)')
        return cb

    def plot_state_cbound(
        self,
        this_state,
        *args,
        ax: plt.Axes | None = None,
        smoother: bool = False,
        cb_fac: float = 3.,
        **kwargs
    ) -> None:
        """Plot the time history of state estimates"""
        ax = plt.gca() if ax is None else ax
        tvec = self.get_time_elapsed()
        xvec = self.get_mean(this_state, smoother=smoother)
        xvec_var = self.get_cov(this_state, smoother=smoother)
        cb_width = np.ravel(cb_fac * xvec_var**0.5)
        cb = ax.fill_between(tvec, xvec - cb_width, xvec +
                             cb_width, *args, **kwargs)
        # ax.set_ylabel(f'{self.state_names[state_index]}')
        ax.set_xlim([tvec[0], tvec[-1]])
        # ax.set_xlabel('Time elapsed (Seconds)')
        return cb

    # def plot(
    #     self,
    #     smoother: bool = False,
    #     fig_size=(6, 6)
    # ) -> None:
    #     """Plot the time history of state estimates"""
    #     self.nx
    #     fig, ax = plt.subplots(, figsize=fig_size)
    #     ax = plt.gca() if ax is None else ax
    #     tvec = self.get_time_elapsed()
    #     xvec = self.get_mean(this_state, smoother=smoother)
    #     cb = ax.plot(tvec, xvec, *args, **kwargs)
    #     # ax.set_ylabel(f'{self.state_names[state_idx]}')
    #     ax.set_xlim([tvec[0], tvec[-1]])
    #     # ax.set_xlabel('Time elapsed (Seconds)')
    #     return cb

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
        self.check_state_index(x_indx)
        self.check_state_index(y_indx)
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
        self.check_state_index(x_indx)
        self.check_state_index(y_indx)
        ax = plt.gca() if ax is None else ax
        xlocs = self.get_mean(x_indx, smoother=smoother)
        ylocs = self.get_mean(y_indx, smoother=smoother)
        cb = ax.plot(xlocs, ylocs, *args, **kwargs)
        return cb

    def plot_metric(
        self,
        ax: plt.Axes | None = None,
        metric: str = 'FilterNIS',
        **kwargs
    ) -> None:
        """Plot the time history of a performance metric"""
        ax = plt.gca() if ax is None else ax
        if metric not in self.df.columns:
            self.raiseit(f'Invalid metric name {metric}')
        dfshort = self.df[~self.df[metric].isna()]
        tvec = dfshort['TimeElapsed'].values
        xvec = dfshort[metric].values
        ax.plot(tvec, xvec, **kwargs)
        ax.set_ylabel(metric)
        ax.set_xlim([tvec[0], tvec[-1]])
        ax.set_xlabel('Time elapsed (Seconds)')

### Matrix/vector handling###

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f'----{self.name}-{self.id}----\n'
        out_str += f'State labels     : ' + ','.join(self.state_names) + '\n'
        out_str += f'State dimension  : {self._nx}\n'
        out_str += f'Observation dim  : {self._ny}\n'
        out_str += f'Time interval dt : {self.dt}\n'
        out_str += f'Tolerance in dt  : {self.dt_tol}\n'
        return out_str

    def check_state_index(self, indx: int):
        """Check the validity of the state index"""
        if indx > self.nx:
            self.raiseit(f'Invalid {indx}, choose state_index < {self.nx}')

    def check_state_name(self, iname: str):
        """Check the validity of the state name"""
        if iname not in self.state_names:
            self.raiseit(
                f'Invalid state {iname}, choose from {self.state_names}')

    def mat_setter(self, in_mat, to_shape=None) -> ndarray:
        """Returns a valid numpy array2d while checking for its shape"""
        in_mat = np.atleast_2d(np.asarray_chkfinite(in_mat, dtype=float))
        if in_mat.ndim != 2:
            self.raiseit(f'Need 2d array, input dim: {in_mat.ndim}')
        if to_shape is not None:
            if in_mat.shape != to_shape:
                print('Shape mismatch!')
                self.raiseit(f'Required: {to_shape}, Input: {in_mat.shape}')
        return in_mat

    def vec_setter(self, in_vec, to_size=None) -> ndarray:
        """Returns a valid numpy array1d while checking for its shape"""
        in_vec = np.atleast_1d(np.asarray_chkfinite(in_vec, dtype=float))
        in_vec = in_vec.flatten()
        if to_size is not None:
            if in_vec.size != to_size:
                print('Size mismatch!')
                self.raiseit(f'Required: {to_size}, Input: {in_vec.size}')
        return in_vec

    def float_setter(self, in_val) -> float:
        """Return a valid scalar"""
        in_val = np.asarray_chkfinite(in_val, dtype=float)
        if in_val.size != 1:
            self.raiseit(f'Need scalar!, change input:{in_val} to scalar')
        return float(in_val.item())

    def int_setter(self, in_val) -> int:
        """Return a valid scalar"""
        in_val = np.asarray_chkfinite(in_val, dtype=int)
        if in_val.size != 1:
            self.raiseit(f'Need scalar!, change input:{in_val} to scalar')
        return int(in_val.item())

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.name}: {outstr}')


### Getter for private class variables at last update/forecast###


    @ property
    def nx(self) -> int:
        """Dimension of state space"""
        return self._nx

    @ property
    def ny(self) -> int:
        """Dimension of observation space"""
        return self._ny

    @ property
    def m(self) -> ndarray:
        """State mean"""
        return self._m

    @ property
    def P(self) -> ndarray:
        """State covariance matrix"""
        return self._P

    @property
    def lifespan(self):
        """Returns the time duration of existence till the last update"""
        return self.last_update_at - self.get_time_elapsed()[0]

    @ property
    def time_elapsed(self) -> float:
        """Time elapsed so far """
        return self._time_elapsed

    @ property
    def last_update_at(self) -> float:
        """Time elapsed so far """
        return self._last_update_at

    @ property
    def nis(self) -> float:
        """Normalized innovation squared at the last update"""
        return self._nis

    @ property
    def nees(self) -> float:
        """Normalized estimation error squared at the last update"""
        return self._nees

    @ property
    def loglik(self) -> float:
        """Normalized estimation error squared at the last update"""
        return self._loglik

    @ property
    def history(self) -> pd.DataFrame:
        """History of the filter in the form of a dictionary"""
        return self._history

    @ property
    def df(self) -> pd.DataFrame:
        """History of the filter/smoother in the form of a pandas dataframe"""
        return pd.DataFrame(self.history)

    @ property
    def df_filter(self) -> pd.DataFrame:
        """History of the filter in the form of a pandas dataframe"""
        smoother = False
        dff = self.df.loc[:, ['TimeElapsed']].copy()
        for iname in self.state_names:
            dff[iname] = self.get_mean(iname, smoother=smoother)
            dff[iname + '_var'] = self.get_cov(iname, smoother=smoother)
        dff['Observation'] = self.df['Observation']
        dff['Training'] = True
        dff.loc[self.df['Observation'].isna(), 'Training'] = False
        dff[self.id_col] = self.id
        for cname in self.filter_metrics:
            dff[cname] = self.df[cname]
        # dff[dff.select_dtypes(np.float64).columns] = dff.select_dtypes(
        #     np.float64).astype(np.float32)
        return dff

    @ property
    def df_smoother(self) -> pd.DataFrame:
        """History of the filter in the form of a pandas dataframe"""
        if 'SmootherMean' not in self.df.columns:
            self.raiseit('No smoother history found. Run smoother first!')
        smoother = True
        dff = self.df.loc[:, ['TimeElapsed']].copy()
        for iname in self.state_names:
            dff[iname] = self.get_mean(iname, smoother=smoother)
            dff[iname + '_var'] = self.get_cov(iname, smoother=smoother)
        dff['Training'] = True
        dff.loc[self.df['Observation'].isna(), 'Training'] = False
        dff[self.id_col] = self.id
        for cname in self.smoother_metrics:
            dff[cname] = self.df[cname]
        # dff[dff.select_dtypes(np.float64).columns] = dff.select_dtypes(
        #     np.float64).astype(np.float32)
        # dff[dff.select_dtypes(np.int64).columns] = dff.select_dtypes(
        #     np.int64).astype(np.int32)
        return dff

### Getter/Setter for Truth and observations###

    @ property
    def truth(self) -> ndarray:
        """State truth"""
        return self._truth

    @ truth.setter
    def truth(self, in_mat: ndarray | None) -> None:
        """Setter for state truth"""
        self._truth = self.vec_setter(
            in_mat, self.nx) if in_mat is not None else None

    @ property
    def obs(self) -> ndarray:
        """Observation vector"""
        return self._obs

    @ obs.setter
    def obs(self, in_mat: ndarray | None) -> None:
        """Setter for observation vector"""
        self._obs = self.vec_setter(
            in_mat, self.ny) if in_mat is not None else None

    @ property
    def state_names(self) -> list:
        """Labels"""
        return self._state_names

    @ state_names.setter
    def state_names(self, in_val: Sequence[str]) -> None:
        """Setter for labels"""
        if len(in_val) != self.nx:
            self.raiseit(f'Number of labels should be {self.nx}')
        self._state_names = [ix for ix in in_val]


### Getter/Setter for matrices of dynamics model###

    @ property
    def F(self) -> ndarray:
        """State transition matrix"""
        return self._F

    @ F.setter
    def F(self, in_mat: ndarray) -> None:
        """Setter for state transition matrix"""
        self._F = self.mat_setter(in_mat, (self.nx, self.nx))

    @ property
    def G(self) -> ndarray:
        """Model error Jacobian matrix"""
        return self._G

    @ G.setter
    def G(self, in_mat: ndarray) -> None:
        """Setter for state transition matrix"""
        self._G = self.mat_setter(in_mat, (self.nx, self.nx))

    @ property
    def Q(self) -> ndarray:
        """Process error covariance matrix"""
        return self._Q

    @ Q.setter
    def Q(self, in_mat: ndarray) -> None:
        """Setter for process error covariance matrix"""
        self._Q = self.mat_setter(in_mat, (self.nx, self.nx))

    @ property
    def qbar(self) -> ndarray:
        """Process error mean"""
        return self._qbar

    @ qbar.setter
    def qbar(self, in_mat: ndarray) -> None:
        """Setter for process error mean"""
        self._qbar = self.vec_setter(in_mat, self.nx)

### Getter/Setter for matrices of observation model###

    @ property
    def H(self) -> ndarray:
        """observation matrix"""
        return self._H

    @ H.setter
    def H(self, in_mat: ndarray) -> None:
        """Setter for observation-state relation"""
        self._H = self.mat_setter(in_mat, (self.ny, self.nx))

    @ property
    def J(self) -> ndarray:
        """observation error jacobian matrix"""
        return self._J

    @ J.setter
    def J(self, in_mat: ndarray) -> None:
        """Setter for observation-state relation"""
        self._J = self.mat_setter(in_mat, (self.ny, self.ny))

    @ property
    def R(self) -> ndarray:
        """observation error covariance matrix"""
        return self._R

    @ R.setter
    def R(self, in_mat: ndarray) -> None:
        """Setter for observation error covariance matrix"""
        self._R = self.mat_setter(in_mat, (self.ny, self.ny))

    @ property
    def rbar(self) -> ndarray:
        """Observation error mean"""
        return self._rbar

    @ rbar.setter
    def rbar(self, in_mat: ndarray) -> None:
        """Setter for obs error mean"""
        self._rbar = self.vec_setter(in_mat, self.ny)

    @ staticmethod
    def symmetrize(in_mat: ndarray) -> ndarray:
        """Return a symmetrized version of NumPy array"""
        # if np.any(np.isnan(in_mat)) or np.any(in_mat.diagonal() < 0.):
        #     print('\np update went wrong!')
        #     print(in_mat.diagonal())
        return (in_mat + in_mat.T) / 2.


def get_covariance_ellipse(icov: ndarray, fac: float):
    """Retruns width, height, and angle of the covariance ellipse"""
    vals, vecs = np.linalg.eigh(icov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2. * fac * np.sqrt(vals)
    return width, height, theta
