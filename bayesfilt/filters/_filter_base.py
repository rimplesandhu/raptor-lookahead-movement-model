""" Kalman filter base class """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from ._logger import FilterLogger
from ._metrics import FilterMetrics
from ._variables import FilterVariables
from .utils import assign_mat

import sys
from dataclasses import dataclass, field
from abc import abstractmethod
from copy import deepcopy
import numpy as np
from numpy import ndarray
from typing import Callable
from functools import partial
import numpy as np
from numpy import ndarray
from tqdm import tqdm
TypeFunc21 = Callable[[ndarray, ndarray], ndarray]


@dataclass
class KalmanFilterBase():
    """ Base class for implementing various versions of Kalman Filters"""
    nx: int
    ny: int
    dt: float
    dt_tol: float | None = None
    epsilon: float = 1e-6
    verbose: bool = False
    xnames: list[str] = field(default_factory=list, repr=False)

    #  add/subtract functions (to handle angles)
    fun_subtract_x: TypeFunc21 = field(
        default=np.subtract,
        repr=False
    )
    fun_subtract_y: TypeFunc21 = field(
        default=np.subtract,
        repr=False
    )
    fun_weighted_mean_x: TypeFunc21 = field(
        default=partial(np.average, axis=0),
        repr=False
    )
    fun_weighted_mean_y: TypeFunc21 = field(
        default=partial(np.average, axis=0),
        repr=False
    )

    def __post_init__(self):
        """post initiation function"""
        # tracker, metrics and logger
        self.vars = FilterVariables()  # time-varying variables
        self.metrics = FilterMetrics()  # metrics
        self.logger = FilterLogger()  # filter logger
        self.slogger = FilterLogger()  # smoother logger

        # default values
        if self.dt_tol is None:
            self.dt_tol = self.dt / 2.
            self.printit(f'Setting dt_tol to {self.dt_tol}')

        # state names
        if self.xnames:
            assert len(self.xnames) == self.nx, 'len(xnames) should be nx!'
        else:
            self.xnames = [f'X{i}' for i in range(self.nx)]

    def printit(self, istr):
        """print it"""
        print(f'{self.__class__.__name__}: {istr}', flush=True)

    def reset(self):
        """reset to default"""
        self.vars.reset()
        self.logger.reset()
        self.metrics.reset()
        self.slogger.reset()

    def log_this_timestep(self, smoother: bool = False) -> None:
        """Store this forecast/update step"""
        logger = self.slogger if smoother else self.logger
        logger.record(
            time_elapsed=self.vars.t,
            state_mean=self.vars.m,
            state_cov=self.vars.P,
            obs=self.vars.y,
            obs_cov=self.vars.R,
            metrics=self.metrics.as_dict(),
            flag=self.vars.flag
        )

    def initiate(
        self,
        t0: float,
        m0: ndarray,
        P0: ndarray,
        flag: str = 'Start'
    ) -> None:
        """Initiate the filter"""
        # first reset everything
        self.printit(f'Initiating filter at {np.round(t0, 3)} sec..')
        self.reset()
        self.vars.t_start = t0
        self.vars.flag = flag

        # update variable tracker
        self.vars.t = t0
        self.vars.m = assign_mat(m0, self.nx)
        self.vars.P = assign_mat(P0, (self.nx, self.nx))

        # log this
        self.log_this_timestep()

    def forecast_upto(
        self,
        upto_time: float,
        flag: str | None = None
    ) -> None:
        """forecast step upto some time in future"""
        time_diff = upto_time - self.vars.t
        if abs(time_diff) > self.dt_tol:
            if time_diff < 0.:
                istr = f'{self.vars.t}->{upto_time}'
                self.raiseit(f'Forecasting backward in time:{istr}!')
            nsteps = int(np.round(time_diff / self.dt))
            for _ in range(nsteps):
                self.forecast(flag=flag)

    @abstractmethod
    def forecast(self, flag: str | None = None) -> None:
        """Forecast step"""

    @abstractmethod
    def update(
        self,
        obs_y: ndarray,
        obs_R: ndarray,
        obs_flag: str | None
    ) -> None:
        """Update step"""

    def forecast_postprocess(self):
        """postrpocessing step for forecast"""
        self.vars.t += self.dt
        self.vars.y = None
        self.vars.R = None
        self.vars.mres = None
        self.vars.yres = None
        self.vars.Sinv = None
        self.metrics.compute()
        self.log_this_timestep()

    def update_postprocess(self):
        """postprocessing step for update"""
        self.metrics.compute(
            residual_x=self.vars.mres,
            residual_y=self.vars.yres,
            precision_x=self.vars.Pinv,
            precision_y=self.vars.Sinv
        )
        self.t_last_update = deepcopy(self.vars.t)
        self.log_this_timestep()

    def filter(
        self,
        list_of_t: list[float],
        list_of_y: list[ndarray],
        list_of_R: list[ndarray] | ndarray,
    ) -> None:
        """Run filtering assuming F, H, Q matrices are time invariant"""
        # make sure the provided data is compatible
        nsteps = len(list_of_t)
        assert len(list_of_y) == nsteps, 'Length mismatch: list_of_y/list_of_t'

        # see if obs cov matrix is time-invariant and conforming
        if not isinstance(list_of_R, ndarray):
            assert len(list_of_R) == nsteps, 'Len mismatch: list_of_R/list_of_t'

        # make sure state is initiated
        assert self.vars.is_initiated(), 'Need to initiate state first!'

        # run loop
        tloop = tqdm(
            iterable=enumerate(list_of_t),
            total=nsteps,
            desc=self.__class__.__name__,
            leave=True,
            file=sys.stdout
        )
        for ith, itime in tloop:
            # make sure the observations arrive ahead of current time
            istr = f'y at {itime}, x at {self.vars.t}'
            assert self.vars.t - itime < self.dt_tol, istr

            # forecast upto the next obs
            if itime - self.vars.t >= self.dt:
                self.forecast_upto(itime, flag='Forecast')

            # update
            if isinstance(list_of_R, ndarray):
                Rmat = list_of_R
            else:
                Rmat = list_of_R[ith]

            self.update(
                obs_y=list_of_y[ith],
                obs_R=Rmat,
                obs_flag='Update'
            )

    @abstractmethod
    def _backward_filter(self, smean_next, scov_next) -> None:
        """Backward filter"""

    def backward_update_postprocess(self):
        """Backward update postprocess"""
        pass

    def smoother(self):
        """Run smoothing assuming model/measurement eq are time invariant"""
        # check if filter data exists
        nsteps = len(self.logger.time_elapsed)
        if nsteps < 2:
            raise ValueError('No state history found, run filter() first!')

        # reset the smoother
        self.slogger.reset()
        self.metrics.reset()

        # run loop
        tloop = tqdm(
            iterable=reversed(range(nsteps)),
            total=nsteps,
            desc=self.__class__.__name__ + '(S)',
            leave=True,
            file=sys.stdout
        )
        cur_t = self.logger.time_elapsed[-1] + self.dt
        for i in tloop:
            self.vars.t = deepcopy(self.logger.time_elapsed[i])
            time_diff = cur_t - self.vars.t
            if abs(time_diff) > self.dt_tol:
                self.vars.m = deepcopy(self.logger.state_mean()[i])
                self.vars.P = deepcopy(self.logger.state_var()[i])
                self.vars.y = deepcopy(self.logger.obs(i))
                self.vars.R = deepcopy(self.logger.obs_var(i))
                self.vars.flag = deepcopy(self.logger.flag(i))
                if i != nsteps - 1:
                    self.backward_update(smoothed_m, smoother_P)
                smoothed_m = deepcopy(self.vars.m)
                smoother_P = deepcopy(self.vars.P)
                cur_t = deepcopy(self.vars.t)
                self.log_this_timestep(smoother=True)
        self.slogger.reverse()


#  def smoother(self):
#         """Run smoothing assuming model/measurement eq are time invariant"""
#         # check if filter data exists
#         nsteps = len(self.logger.time_elapsed)
#         if nsteps < 2:
#             raise ValueError('No state history found, run filter() first!')

#         # reset the smoother
#         self.slogger.reset()
#         self.smetrics.reset()
#         self.smetrics.compute()

#         # run loop
#         tloop = tqdm(
#             iterable=reversed(range(nsteps)),
#             total=nsteps,
#             desc=self.__class__.__name__ + '(S)',
#             leave=True,
#             file=sys.stdout
#         )
#         for i in tloop:
#             self.m = deepcopy(self.logger.state_mean()[i])
#             self.P = deepcopy(self._dfraw[self.cov_colname][i])
#             self.y = deepcopy(self._dfraw[self.y_colname][i])
#             self.R = deepcopy(self._dfraw[self.ycov_colname][i])
#             if i != nsteps - 1:
#                 smean_next = self._dfraw[self.smean_colname][-1]
#                 scov_next = self._dfraw[self.scov_colname][-1]
#                 self._backward_filter(smean_next, scov_next)
#             self._dfraw[self.smean_colname].append(deepcopy(self.m))
#             self._dfraw[self.scov_colname].append(deepcopy(self.P))
#             self._dfraw[self.smetrics_colname].append(self.cur_smetrics)
#         for cname in colnames:
#             self._dfraw[cname].reverse()

 # def filter(
    #     self,
    #     list_of_time: Sequence[float],
    #     list_of_obs: Sequence[ndarray],
    #     list_of_R: Sequence[ndarray] | None = None,
    # ) -> None:
    #     """Run filtering assuming F, H, Q matrices are time invariant"""

    #     self.initiate()
    #     nsteps = len(list_of_time)
    #     if len(list_of_obs) != nsteps:
    #         istr = 'Length mismatch between list_of_time and list_of_obs'
    #         self.raiseit(f'{istr}: {nsteps} vs {len(list_of_obs)}')
    #     if list_of_R is not None:
    #         if len(list_of_R) != nsteps:
    #             istr = 'Length mismatch between list_of_time and list_of_R'
    #             self.raiseit(f'{istr}: {nsteps} vs {len(list_of_R)}')
    #     else:
    #         if self.R is None:
    #             self.raiseit('Need to initiate obs cov matrix R')

    #     tloop = enumerate(list_of_time)
    #     if self.verbose:
    #         tloop = tqdm(enumerate(list_of_time), total=len(list_of_time),
    #                      desc=self.__class__.__name__)
    #     for k, itime in tloop:
    #         if self.time_elapsed - itime > self.dt_tol:
    #             istr = f'y at {itime}, x at {self.time_elapsed}'
    #             self.raiseit(f'Skipping observation, {istr}!')
    #         if itime - self.time_elapsed >= self.dt:
    #             self.forecast_upto(itime)
    #             self.P = sym_posdef_matrix(self.P)
    #         self.y = list_of_obs[k]
    #         if list_of_R is not None:
    #             self.R = list_of_R[k]
    #         self.update()

    # def get_loglik_of_obs(
    #     self,
    #     y_obs: ndarray,
    #     ignore_obs_inds: list[int] | None = None
    # ) -> None:
    #     """
    #     Compute log-likelihood of this observation
    #     """
    #     y_pred = self.mat_H @ self.m
    #     _residual = y_obs - y_pred
    #     _this_smat = self.mat_H @ self.P @ self.mat_H.T + self.R
    #     if ignore_obs_inds is not None:
    #         for idx in ignore_obs_inds:
    #             _residual[idx] = 0.
    #     _s_inv = np.linalg.pinv(_this_smat, hermitian=True)
    #     # _s_inv = np.linalg.inv(_this_smat)
    #     # print(_s_inv.diagonal())
    #     this_loglik = self.ny * np.log(2. * np.pi)
    #     this_loglik += np.log(np.linalg.det(_this_smat))
    #     this_loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
    #     this_loglik = np.linalg.multi_dot([_residual.T, _s_inv, _residual])
    #     # this_loglik = np.dot(_residual, np.dot(_s_inv, _residual))
    #     # this_loglik = np.dot(_residual, _residual)
    #     this_loglik *= -0.5
    #     return this_loglik

    # def initiate_state(
    #     self,
    #     dict_of_mean_std: dict[str, tuple[float, float]]
    # ) -> None:
    #     """Initiate the time, state mean and covariance matrix"""
    #     self.m = np.zeros((self.nx,))
    #     self.P = np.eye(self.nx)
    #     for i, iname in enumerate(self.state_names):
    #         if iname in dict_of_mean_std.keys():
    #             self.m[i] = dict_of_mean_std[iname][0]
    #             self.P[i, i] = dict_of_mean_std[iname][1]**2

 # def get_state_index(self, state: int | str) -> int:
    #     """Get state index and name"""
    #     if isinstance(state, int):
    #         if state > self.nx:
    #             self.raiseit(f'Invalid index {state}, choose < {self.nx}')
    #         idx = state
    #     elif isinstance(state, str):
    #         if state not in self.state_names:
    #             self.raiseit(f'Invalid {state}, Valid: {self.state_names}')
    #         idx = self.state_names.index(state)
    #     return idx

    # def get_mean(
    #     self,
    #     state: int | str,
    #     smoother: bool = False
    # ) -> ndarray:
    #     """Get time series of state mean for a given index/name of the state"""
    #     colname = self.smean_colname if smoother else self.mean_colname
    #     if colname not in self.dfraw.columns:
    #         self.raiseit('No smoother history found, run smoother() first!')
    #     idx = self.get_state_index(state)
    #     return np.stack(self.dfraw[colname].values)[:, idx]

    # def get_cov(
    #     self,
    #     state: str | int | tuple[int, int] | tuple[str, str],
    #     smoother: bool = False
    # ) -> ndarray:
    #     """Get state covariance matrix for the desired indices"""
    #     cname = self.scov_colname if smoother else self.cov_colname
    #     if isinstance(state, int) | isinstance(state, str):
    #         idx = self.get_state_index(state)
    #         idx_pair = [idx, idx]
    #     elif isinstance(state, Sequence):
    #         assert len(state) == 2, 'Need a pair of state idx/names!'
    #         idx_pair = [0, 0]
    #         for i, idx in enumerate(state):
    #             idx_pair[i] = self.get_state_index(idx)
    #     else:
    #         self.raiseit('Need either int or str or (int,int) or (str,str)')
    #     return np.stack(self.dfraw[cname].values)[:, idx_pair[0], idx_pair[1]]

    # def get_df(
    #     self,
    #     smoother=False,
    #     variance: bool = False,
    #     metrics: bool = True
    # ) -> pd.DataFrame:
    #     """Returns short version of pandas dataframe"""
    #     out_df = self.dfraw.loc[:, [self.time_colname]].copy()
    #     mcol = self.smetrics_colname if smoother else self.metrics_colname
    #     for iname in self.state_names:
    #         out_df[iname] = self.get_mean(iname, smoother=smoother)
    #         if variance:
    #             out_df[iname + '_var'] = self.get_cov(iname, smoother=smoother)
    #     # dff['Observation'] = self.dfraw['Observation']
    #     out_df['ObjectId'] = self.objectid
    #     if metrics:
    #         _mdf = self.dfraw[mcol].apply(pd.Series)
    #         out_df = pd.concat([out_df, _mdf], axis=1)
    #     return out_df

    # @property
    # def df(self) -> pd.DataFrame:
    #     """History of the filter in the form of a pandas dataframe"""
    #     return self.get_df(smoother=False)

    # @property
    # def dfs(self) -> pd.DataFrame:
    #     """History of the filter in the form of a pandas dataframe"""
    #     return self.get_df(smoother=True)

    # def plot_state_mean(
    #     self,
    #     state,
    #     *args,
    #     ax: plt.Axes | None = None,
    #     smoother: bool = False,
    #     **kwargs
    # ) -> None:
    #     """Plot the time history of state estimates"""
    #     ax = plt.gca() if ax is None else ax
    #     xvec = self.get_mean(state, smoother=smoother)
    #     cb = ax.plot(self.tlist, xvec, *args, **kwargs)
    #     state_idx = self.get_state_index(state)
    #     ax.set_ylabel(f'{self.state_names[state_idx]}')
    #     ax.set_xlim([self.tlist[0], self.tlist[-1]])
    #     ax.set_xlabel('Time elapsed (Seconds)')
    #     return cb

    # def plot_state_cbound(
    #     self,
    #     state,
    #     *args,
    #     ax: plt.Axes | None = None,
    #     smoother: bool = False,
    #     cb_fac: float = 3.,
    #     **kwargs
    # ) -> None:
    #     """Plot the time history of state estimates"""
    #     ax = plt.gca() if ax is None else ax
    #     xvec = self.get_mean(state, smoother=smoother)
    #     xvec_var = self.get_cov(state, smoother=smoother)
    #     cb_width = np.ravel(cb_fac * xvec_var**0.5)
    #     cb = ax.fill_between(self.tlist, xvec - cb_width, xvec +
    #                          cb_width, *args, **kwargs)
    #     state_idx = self.get_state_index(state)
    #     ax.set_ylabel(f'{self.state_names[state_idx]}')
    #     ax.set_xlim([self.tlist[0], self.tlist[-1]])
    #     ax.set_xlabel('Time elapsed (Seconds)')
    #     return cb

    # def plot_trajectory_cbound(
    #     self,
    #     x_indx: int,
    #     y_indx: int,
    #     ax: plt.Axes | None = None,
    #     smoother: bool = False,
    #     cb_fac: float = 3.,
    #     **kwargs
    # ) -> None:
    #     """Plot the trajectory"""
    #     ax = plt.gca() if ax is None else ax
    #     xlocs = self.get_mean(x_indx, smoother=smoother)
    #     ylocs = self.get_mean(y_indx, smoother=smoother)
    #     xy_vars = self.get_cov([x_indx, y_indx], smoother=smoother)
    #     for xloc, yloc, icov in zip(xlocs, ylocs, xy_vars):
    #         width, height, angle = get_covariance_ellipse(icov, cb_fac)
    #         ellip = Ellipse(xy=[xloc, yloc], width=width, height=height,
    #                         angle=angle, **kwargs)
    #         ax.add_artist(ellip)

    # def plot_trajectory_mean(
    #     self,
    #     x_indx: int,
    #     y_indx: int,
    #     *args,
    #     ax: plt.Axes | None = None,
    #     smoother: bool = False,
    #     **kwargs
    # ) -> None:
    #     """Plot the trajectory"""
    #     ax = plt.gca() if ax is None else ax
    #     xlocs = self.get_mean(x_indx, smoother=smoother)
    #     ylocs = self.get_mean(y_indx, smoother=smoother)
    #     cb = ax.plot(xlocs, ylocs, *args, **kwargs)
    #     return cb

    # @property
    # def metrics(self) -> dict[str, float]:
    #     """Return the summary metrics"""
    #     idf = self.df
    #     metric_names = list(self.cur_metrics.keys())
    #     return idf.loc[~idf[metric_names[0]].isna(), metric_names].sum()

    # @property
    # def smetrics(self) -> dict[str, float]:
    #     """Return the summary metrics"""
    #     idf = self.dfs
    #     metric_names = list(self.cur_smetrics.keys())
    #     return idf.loc[~idf[metric_names[0]].isna(), metric_names].sum()

    # def plot_metric(
    #     self,
    #     metric: str = 'NIS',
    #     smoother=False,
    #     ax: plt.Axes | None = None,
    #     **kwargs
    # ) -> None:
    #     """Plot the time history of a performance metric"""
    #     ax = plt.gca() if ax is None else ax
    #     mnames = list(self.cur_metrics.keys())
    #     if metric not in mnames:
    #         self.raiseit(f'Invalid metric {metric}, choose from {mnames}')
    #     idf = self.dfs if smoother else self.df
    #     idfshort = idf[~idf[metric].isna()]
    #     tvec = idfshort[self.time_colname].values
    #     xvec = idfshort[metric].values
    #     ax.plot(tvec, xvec, **kwargs)
    #     ax.set_ylabel(metric)
    #     ax.set_xlim([tvec[0], tvec[-1]])
    #     ax.set_xlabel('Time elapsed (Seconds)')

   # def compute_metrics(
    #     self,
    #     xres: ndarray | None = None,
    #     xprec: ndarray | None = None,
    #     yres: ndarray | None = None,
    #     yprec: ndarray | None = None
    # ):
    #     """compute filter performance metrics"""
    #     idict = {
    #         'MetricXresNorm': np.nan,
    #         'MetricYresNorm': np.nan,
    #         'MetricNIS': np.nan,
    #         'MetricNEES': np.nan,
    #         'MetricLogLik': np.nan
    #     }
    #     if (yres is not None) and (yprec is not None):
    #         idict['MetricYresNorm'] = np.linalg.norm(yres)
    #         idict['MetricNIS'] = np.linalg.multi_dot([yres.T, yprec, yres])
    #         prec_det = np.linalg.det(yprec)
    #         idict['MetricLogLik'] = -0.5 * (self.ny * np.log(2. * np.pi) -
    #                                         np.log(prec_det) + idict['MetricNIS'])
    #     if (xres is not None) and (xprec is not None):
    #         idict['MetricXresNorm'] = np.linalg.norm(xres)
    #         idict['MetricNEES'] = np.linalg.multi_dot([xres.T, xprec, xres])
    #     return idict

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
