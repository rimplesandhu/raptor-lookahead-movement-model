""" Commonly used functions """
from abc import ABC, abstractmethod
from typing import Callable
from matplotlib.patches import Ellipse
import numpy as np
from numpy import ndarray
from copy import deepcopy
# pylint: disable=invalid-name

clrs = ['#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']

Func2to1 = Callable[[ndarray, ndarray], ndarray]
Func1to1 = Callable[[ndarray], ndarray]


def check_mat(
    in_mat: ndarray,
    in_shape: tuple | int | None = None,
    error: Exception = ValueError
) -> ndarray:
    """Returns a valid numpy array while checking for its shape"""
    in_mat = np.atleast_1d(np.asarray_chkfinite(in_mat, dtype=float))
    if in_shape is not None:
        in_shape = (in_shape,) if isinstance(in_shape, int) else in_shape
        if in_mat.shape != in_shape:
            istr = f'Required:{in_shape}, Input:{in_mat.shape}'
            raise error(istr)
    return in_mat


def symmetrize_mat(
    in_mat: ndarray,
    eps: float
) -> ndarray:
    """Return a symmetrized, regularized covariance matrix"""
    nrow, ncol = np.shape(in_mat)
    assert nrow == ncol, 'symmetrize_mat requires a symmetric matrix!'
    return (in_mat + in_mat.T) / 2. + np.diag([eps] * nrow)


def subtract_func(
    x1: ndarray,
    x2: ndarray,
    angle_index: int | None = None,
    radians: bool = False
) -> ndarray:
    """Subtract two vectors that include angle"""
    # x0 = self.matrix(x0, self.nx)
    # x1 = self.matrix(x1, self.nx)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    ival = np.pi if radians else 180.
    assert x1.size == x2.size, 'Shape mismatch!'
    xres = np.subtract(x1, x2)
    if angle_index is not None:
        # angle_index = self.scaler(angle_index, dtype='int32')
        angle_index = int(angle_index)
        assert angle_index < x1.size, 'Invalid angle index!'
        # xres[angle_index] = np.mod(xres[angle_index], 2.0 * np.pi)
        if xres[angle_index] > ival:
            xres[angle_index] -= 2. * ival
        if xres[angle_index] <= -ival:
            xres[angle_index] += 2. * ival
        # xres[angle_index] = angle_correction(xres[angle_index])
    return xres


def angle_correction(iangle):
    """Correct angles to be between -pi,pi"""
    new_angle = iangle
    if iangle > np.pi:
        new_angle = iangle - 2. * np.pi
    elif iangle <= -np.pi:
        new_angle = iangle + 2. * np.pi
    return new_angle


def mean_func(
    list_of_vecs: list[ndarray],
    weights: list[float] | None = None,
    angle_index: int | None = None,
    radians: bool = False
) -> ndarray:
    """Returns means of vectors with wgts while handling angles"""
    assert isinstance(list_of_vecs, list), 'Need a list of numpy arrays'
    if weights is None:
        weights = [1. / len(list_of_vecs)] * len(list_of_vecs)
    if len(list_of_vecs) != len(weights):
        raise ValueError('Size mismatch between vecs and wgts!')
    yvec = sum([iw * iy for iw, iy in zip(weights, list_of_vecs)])
    if angle_index is not None:
        angle_index = int(angle_index)
        angles = np.array([ivec[angle_index] for ivec in list_of_vecs])
        if not radians:
            angles = np.radians(angles)
        # print(np.degrees(angles))
        # print(np.sin(angles))
        # print(np.cos(angles))
        # print(np.multiply(np.sin(angles), weights))
        # print(np.multiply(np.cos(angles), weights))
        # angles = np.vectorize(angle_correction)(angles)
        # print(np.degrees(angles))
        # print(np.sin(angles))
        # print(np.cos(angles))
        sum_sin = np.dot(np.sin(angles), weights)
        sum_cos = np.dot(np.cos(angles), weights)
        yvec[angle_index] = np.arctan2(sum_sin, sum_cos)
        if not radians:
            yvec[angle_index] = np.degrees(yvec[angle_index])
        # print(weights, sum_sin, sum_cos, yvec)
    return yvec


def get_covariance_ellipse(icov: ndarray, fac: float):
    """Retruns width, height, and angle of the covariance ellipse"""
    vals, vecs = np.linalg.eigh(icov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2. * fac * np.sqrt(vals)
    return width, height, theta


def validate_array(
    in_mat: ndarray,
    in_shape: tuple | int | None = None,
    return_array=False,
    error: Exception = ValueError
) -> ndarray:
    """Returns a valid numpy array while checking for its shape and validity"""
    out_mat = None
    in_mat = np.atleast_1d(np.asarray_chkfinite(in_mat, dtype=float))
    if in_shape is not None:
        in_shape = (in_shape,) if isinstance(in_shape, int) else in_shape
        # in_shape = (in_shape,) if np.isscalar(in_shape) else in_shape
        if in_mat.shape != in_shape:
            out_str = f'Required shape:{in_shape}, Input shape: {in_mat.shape}'
            raise error(out_str)
    out_mat = in_mat if return_array else out_mat
    return out_mat


def sym_posdef_matrix(in_mat: ndarray) -> ndarray:
    """Return a symmetrized version of NumPy array"""
    if np.any(np.isnan(in_mat)):
        raise ValueError('Matrix has nans!')
    if np.any(in_mat.diagonal() < 0.):
        raise ValueError('Matrix has negative entries on diagonal!')
    try:
        _ = np.linalg.cholesky(in_mat)
    except np.linalg.linalg.LinAlgError as _:
        print('Matrix is not pos def!')
        raise
    return (in_mat + in_mat.T) / 2.


# @dataclass
# class FilterAttributesDynamic:
#     """Dynamic attributes of a filter"""
#     objectid: int = 0
#     _y: ndarray | None = None
#     _m: ndarray | None = None
#     _P: ndarray | None = None
#     _R: ndarray | None = None
#     _start_time: float | None = None
#     _current_time: float | None = None
#     _last_update_at: float | None = None

#     # metrics
#     metrics = FilterMetrics()  # filter
#     smetrics = FilterMetrics()  # smoother

#     # logger
#     logger = FilterLogger()  # filter
#     slogger = FilterLogger()  # smoother

#     def store_this_timestep(self, update: bool = False) -> None:
#         """Store this forecast/update step"""
#         if update is True:
#             self._last_update_at = deepcopy(self.time_elapsed)
#             self.logger.record(
#                 time_elapsed=self.time_elapsed,
#                 state_mean=self.m,
#                 state_cov=self.P,
#                 obs=self.y,
#                 obs_cov=self.R,
#                 metrics=self.metrics.as_dict()
#             )

#     # @property
#     # def is_ready_for_smoother(self):
#     #     """Return true if ready for smoother"""
#     #     out_bool = False
#     #     if len(self.dfraw[self.time_colname]) > 1:
#     #         print('No state history found, run filter() first!')
#     #     else:
#     #         out_bool = True
#     #     return out_bool


# TypeFunc21 = Callable[[ndarray, ndarray], ndarray]


# @ dataclass(frozen=True)
# class FilterAttributesStatic:
#     """Static attributes of a filter"""
#     nx: int
#     ny: int
#     dt: float
#     dt_tol: float | None = None
#     epsilon: float = 1e-6
#     verbose: bool = False
#     state_names: list[str] = field(default_factory=list, repr=False)
#     pars: dict[str, float] = field(default_factory=dict, repr=True)

#     # model functions
#     fun_f: TypeFunc21 | None = field(default=None, repr=False)
#     fun_Fjac: TypeFunc21 | None = field(default=None, repr=False)
#     fun_Gjac: TypeFunc21 | None = field(default=None, repr=False)
#     fun_h: TypeFunc21 | None = field(default=None, repr=False)
#     fun_Hjac: TypeFunc21 | None = field(default=None, repr=False)
#     fun_Jjac: TypeFunc21 | None = field(default=None, repr=False)
#     fun_Q: TypeFunc21 | None = field(default=None, repr=False)

#     # matrices
#     mat_F: ndarray | None = field(default=None, repr=False)
#     mat_G: ndarray | None = field(default=None, repr=False)
#     mat_H: ndarray | None = field(default=None, repr=False)
#     mat_J: ndarray | None = field(default=None, repr=False)
#     mat_Q: ndarray | None = field(default=None, repr=False)
#     vec_qbar: ndarray | None = field(default=None, repr=False)
#     vec_rbar: ndarray | None = field(default=None, repr=False)

#     # add/subtract functions
#     fun_subtract_x: TypeFunc21 = field(default=np.subtract, repr=False)
#     fun_subtract_y: TypeFunc21 = field(default=np.subtract, repr=False)
#     fun_weighted_mean_x: TypeFunc21 = field(
#         default=partial(np.average, axis=0),
#         repr=False
#     )
#     fun_weighted_mean_y: TypeFunc21 = field(
#         default=partial(np.average, axis=0),
#         repr=False
#     )

#     def __post_init__(self):
#         """post initiation function"""

#         # default values
#         if self.dt_tol is None:
#             setattr(self, 'dt_tol', self.dt / 2.)
#         if self.vec_qbar is None:
#             setattr(self, 'vec_qbar', np.zeros(self.nx))
#         if self.vec_rbar is None:
#             setattr(self, 'vec_rbar', np.zeros(self.ny))
#         if self.state_names is None:
#             setattr(self, 'state_names',
#                     [f'x_{i}' for i in range(self.nx)])

#         # mat_F takes priority over everything
#         if self.mat_G is None:
#             setattr(self, 'mat_G', np.eye(self.nx))
#             if self.fun_Gjac is None:
#                 setattr(
#                     self, 'fun_Gjac', partial(self.v2m, self.mat_G))

#         if self.mat_J is None:
#             setattr(self, 'mat_J', np.eye(self.ny))
#             if self.fun_Jjac is None:
#                 setattr(
#                     self, 'fun_Jjac', partial(self.v2m, self.mat_J))

#         if (self.mat_F is not None) and (self.fun_f is None):
#             setattr(self, 'fun_f', partial(self.v2v, self.mat_F))
#             setattr(self, 'fun_Fjac', partial(self.v2m, self.mat_F))

#         # mat_H takes priority over everything
#         if (self.mat_H is not None) and (self.fun_h is None):
#             setattr(self, 'fun_h', partial(self.v2v, self.mat_H))
#             setattr(self, 'fun_Hjac', partial(self.v2m, self.mat_H))

#         if self.mat_Q is not None:
#             setattr(self, 'fun_Q', partial(self.v2m, self.mat_Q))

#         self.check_valid_initialization()
#         self.check_valid_matrices()

#     def check_valid_matrices(self):
#         """Check validity of matrices"""
#         if self.mat_F is not None:
#             validate_array(self.mat_F, (self.nx, self.nx))
#         if self.mat_G is not None:
#             validate_array(self.mat_G, (self.nx, self.nx))
#         if self.mat_Q is not None:
#             validate_array(self.mat_Q, (self.nx, self.nx))
#         if self.mat_H is not None:
#             validate_array(self.mat_H, (self.ny, self.nx))
#         if self.mat_J is not None:
#             validate_array(self.mat_J, (self.ny, self.ny))
#         validate_array(self.vec_rbar, (self.ny,))
#         validate_array(self.vec_qbar, (self.nx,))

#     def check_valid_initialization(self):
#         """Check if filter is properly initialized"""
#         if (self.mat_F is None) and (self.fun_f is None):
#             self.raiseit('Either mat_F or fun_f required to initiate filter!')
#         if (self.mat_Q is None) and (self.fun_Q is None):
#             self.raiseit('Either mat_Q or fun_Q required to initiate filter!')
#         if (self.mat_H is None) and (self.fun_h is None):
#             self.raiseit('Either mat_H or fun_h required to initiate filter!')

#     def v2m(
#         self,
#         mat: ndarray | None = None,
#         x: ndarray | None = None,
#         u: ndarray | None = None,
#     ):
#         """dummy funcion that takes in state vector and return a matrix"""
#         return mat

#     def v2v(
#         self,
#         mat: ndarray | None = None,
#         x: ndarray | None = None,
#         u: ndarray | None = None,
#     ):
#         """dummpy funcion that takes in state vector and return a vector"""
#         return mat @ x

#     def raiseit(self, outstr: str = "") -> None:
#         """Raise exception with the out string"""
#         raise ValueError(f'{self.__class__.__name__}: {outstr}')


# class KalmanFilterBase(FilterAttributesStatic, FilterAttributesDynamic):
#     """ Base class for implementing various versions of Kalman Filters"""

#     def __init__(
#         self,
#         *args,
#         **kwargs
#     ):
#         FilterAttributesStatic.__init__(self, *args, **kwargs)
#         FilterAttributesDynamic.__init__(self)

#     def reset(self):
#         """reset to default"""
#         # filter
#         self.logger.reset()
#         self.metrics.reset()

#         # smoother
#         self.slogger.reset()
#         self.smetrics.reset()

#     def initiate(
#         self,
#         t0: float | None = None,
#         m0: ndarray | None = None,
#         P0: ndarray | None = None
#     ) -> None:
#         """Initiate the filter"""
#         if t0 is None:
#             if self.time_elapsed is None:
#                 self.raiseit('Need to initiate time')
#         else:
#             self._start_time = t0
#             self._time_elapsed = t0
#             self._last_update_at = t0
#         if m0 is None:
#             if self.m is None:
#                 self.raiseit('Need to initiate state mean m')
#         else:
#             self.m = m0
#         if P0 is None:
#             if self.P is None:
#                 self.raiseit('Need to initiate state cov P')
#         else:
#             self.P = P0
#         self.reset()
#         self.store_this_timestep()

#     @abstractmethod
#     def forecast(self) -> None:
#         """Forecast step"""
#         self._time_elapsed += self.dt
#         self._y = None
#         self.metrics.compute()

#     @abstractmethod
#     def update(self) -> None:
#         """Update step"""

#     @abstractmethod
#     def _backward_filter(self, smean_next, scov_next) -> None:
#         """Backward filter"""

#     def forecast_upto(
#         self,
#         upto_time: float
#     ) -> None:
#         """Kalman filter forecast step upto some time in future"""
#         time_diff = upto_time - self.time_elapsed
#         if abs(time_diff) > self.dt_tol:
#             if time_diff < 0.:
#                 istr = f'{self.time_elapsed}->{upto_time}'
#                 self.raiseit(f'Forecasting backward in time:{istr}!')
#             nsteps = int(np.round(time_diff / self.dt))
#             for _ in range(nsteps):
#                 self.forecast()

#     def filter(
#         self,
#         list_of_time: Sequence[float],
#         list_of_obs: Sequence[ndarray],
#         list_of_R: Sequence[ndarray] | None = None,
#     ) -> None:
#         """Run filtering assuming F, H, Q matrices are time invariant"""

#         self.initiate()
#         nsteps = len(list_of_time)
#         if len(list_of_obs) != nsteps:
#             istr = 'Length mismatch between list_of_time and list_of_obs'
#             self.raiseit(f'{istr}: {nsteps} vs {len(list_of_obs)}')
#         if list_of_R is not None:
#             if len(list_of_R) != nsteps:
#                 istr = 'Length mismatch between list_of_time and list_of_R'
#                 self.raiseit(f'{istr}: {nsteps} vs {len(list_of_R)}')
#         else:
#             if self.R is None:
#                 self.raiseit('Need to initiate obs cov matrix R')

#         tloop = enumerate(list_of_time)
#         if self.verbose:
#             tloop = tqdm(enumerate(list_of_time), total=len(list_of_time),
#                          desc=self.__class__.__name__)
#         for k, itime in tloop:
#             if self.time_elapsed - itime > self.dt_tol:
#                 istr = f'y at {itime}, x at {self.time_elapsed}'
#                 self.raiseit(f'Skipping observation, {istr}!')
#             if itime - self.time_elapsed >= self.dt:
#                 self.forecast_upto(itime)
#                 self.P = sym_posdef_matrix(self.P)
#             self.y = list_of_obs[k]
#             if list_of_R is not None:
#                 self.R = list_of_R[k]
#             self.update()

#     def smoother(self):
#         """Run smoothing assuming model/measurement eq are time invariant"""
#         nsteps = len(self.dfraw[self.time_colname])
#         if nsteps < 2:
#             self.raiseit('No state history found, run filter() first!')
#         colnames = [self.smean_colname, self.scov_colname,
#                     self.smetrics_colname]
#         for cname in colnames:
#             self._dfraw[cname] = []
#         self.cur_smetrics = self.compute_metrics()
#         tloop = reversed(range(nsteps))
#         if self.verbose:
#             tloop = tqdm(reversed(range(nsteps)), total=nsteps,
#                          desc=self.__class__.__name__ + 'S')
#         for i in tloop:
#             self.m = deepcopy(self._dfraw[self.mean_colname][i])
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

#     def get_loglik_of_obs(
#         self,
#         y_obs: ndarray,
#         ignore_obs_inds: list[int] | None = None
#     ) -> None:
#         """
#         Compute log-likelihood of this observation
#         """
#         y_pred = self.mat_H @ self.m
#         _residual = y_obs - y_pred
#         _this_smat = self.mat_H @ self.P @ self.mat_H.T + self.R
#         if ignore_obs_inds is not None:
#             for idx in ignore_obs_inds:
#                 _residual[idx] = 0.
#         _s_inv = np.linalg.pinv(_this_smat, hermitian=True)
#         # _s_inv = np.linalg.inv(_this_smat)
#         # print(_s_inv.diagonal())
#         this_loglik = self.ny * np.log(2. * np.pi)
#         this_loglik += np.log(np.linalg.det(_this_smat))
#         this_loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
#         this_loglik = np.linalg.multi_dot([_residual.T, _s_inv, _residual])
#         # this_loglik = np.dot(_residual, np.dot(_s_inv, _residual))
#         # this_loglik = np.dot(_residual, _residual)
#         this_loglik *= -0.5
#         return this_loglik

#     def symmetrize(self, in_mat: ndarray) -> ndarray:
#         """Return a symmetrized, regularized covariance matrix"""
#         return (in_mat + in_mat.T) / 2. + np.diag([self.epsilon] * self.nx)

#     @property
#     def y(self) -> ndarray:
#         """Observation vector"""
#         return self._y

#     @property
#     def m(self) -> ndarray:
#         """State mean vector"""
#         return self._m

#     @property
#     def P(self) -> ndarray:
#         """State covariance matrix"""
#         return self._P

#     @property
#     def R(self) -> ndarray:
#         """Observation error covariance matrix"""
#         return self._R

#     @property
#     def get_time_elapsed(self) -> ndarray:
#         """Get time elapsed"""
#         return self.df[self.time_colname].values

#     @property
#     def start_time(self) -> ndarray:
#         """Get time elapsed"""
#         return self._start_time

#     @property
#     def lifespan_to_last_update(self) -> float:
#         """Returns the time duration of existence till the last update"""
#         return self.last_update_at - self.start_time

#     @property
#     def lifespan_to_last_forecast(self) -> float:
#         """Returns the time duration of existence till the last update"""
#         return self.time_elapsed - self.start_time

#     @m.setter
#     def m(self, in_vec: ndarray) -> None:
#         """Setter for m vector"""
#         self._m = validate_array(in_vec, self.nx, return_array=True)

#     @P.setter
#     def P(self, in_mat: ndarray | None) -> None:
#         """Setter for m vector"""
#         self._P = validate_array(in_mat, (self.nx, self.nx), return_array=True)

#     @R.setter
#     def R(self, in_mat: ndarray | None) -> None:
#         """Setter for m vector"""
#         if in_mat is not None:
#             self._R = validate_array(in_mat, (self.ny, self.ny),
#                                      return_array=True)
#         else:
#             self._R = None

#     @y.setter
#     def y(self, in_vec: ndarray | None) -> None:
#         """Setter for m vector"""
#         if in_vec is not None:
#             self._y = validate_array(in_vec, self.ny, return_array=True)
#         else:
#             self._y = None

#     # def initiate_state(
#     #     self,
#     #     dict_of_mean_std: dict[str, tuple[float, float]]
#     # ) -> None:
#     #     """Initiate the time, state mean and covariance matrix"""
#     #     self.m = np.zeros((self.nx,))
#     #     self.P = np.eye(self.nx)
#     #     for i, iname in enumerate(self.state_names):
#     #         if iname in dict_of_mean_std.keys():
#     #             self.m[i] = dict_of_mean_std[iname][0]
#     #             self.P[i, i] = dict_of_mean_std[iname][1]**2

#  # def get_state_index(self, state: int | str) -> int:
#     #     """Get state index and name"""
#     #     if isinstance(state, int):
#     #         if state > self.nx:
#     #             self.raiseit(f'Invalid index {state}, choose < {self.nx}')
#     #         idx = state
#     #     elif isinstance(state, str):
#     #         if state not in self.state_names:
#     #             self.raiseit(f'Invalid {state}, Valid: {self.state_names}')
#     #         idx = self.state_names.index(state)
#     #     return idx

#     # def get_mean(
#     #     self,
#     #     state: int | str,
#     #     smoother: bool = False
#     # ) -> ndarray:
#     #     """Get time series of state mean for a given index/name of the state"""
#     #     colname = self.smean_colname if smoother else self.mean_colname
#     #     if colname not in self.dfraw.columns:
#     #         self.raiseit('No smoother history found, run smoother() first!')
#     #     idx = self.get_state_index(state)
#     #     return np.stack(self.dfraw[colname].values)[:, idx]

#     # def get_cov(
#     #     self,
#     #     state: str | int | tuple[int, int] | tuple[str, str],
#     #     smoother: bool = False
#     # ) -> ndarray:
#     #     """Get state covariance matrix for the desired indices"""
#     #     cname = self.scov_colname if smoother else self.cov_colname
#     #     if isinstance(state, int) | isinstance(state, str):
#     #         idx = self.get_state_index(state)
#     #         idx_pair = [idx, idx]
#     #     elif isinstance(state, Sequence):
#     #         assert len(state) == 2, 'Need a pair of state idx/names!'
#     #         idx_pair = [0, 0]
#     #         for i, idx in enumerate(state):
#     #             idx_pair[i] = self.get_state_index(idx)
#     #     else:
#     #         self.raiseit('Need either int or str or (int,int) or (str,str)')
#     #     return np.stack(self.dfraw[cname].values)[:, idx_pair[0], idx_pair[1]]

#     # def get_df(
#     #     self,
#     #     smoother=False,
#     #     variance: bool = False,
#     #     metrics: bool = True
#     # ) -> pd.DataFrame:
#     #     """Returns short version of pandas dataframe"""
#     #     out_df = self.dfraw.loc[:, [self.time_colname]].copy()
#     #     mcol = self.smetrics_colname if smoother else self.metrics_colname
#     #     for iname in self.state_names:
#     #         out_df[iname] = self.get_mean(iname, smoother=smoother)
#     #         if variance:
#     #             out_df[iname + '_var'] = self.get_cov(iname, smoother=smoother)
#     #     # dff['Observation'] = self.dfraw['Observation']
#     #     out_df['ObjectId'] = self.objectid
#     #     if metrics:
#     #         _mdf = self.dfraw[mcol].apply(pd.Series)
#     #         out_df = pd.concat([out_df, _mdf], axis=1)
#     #     return out_df

#     # @property
#     # def df(self) -> pd.DataFrame:
#     #     """History of the filter in the form of a pandas dataframe"""
#     #     return self.get_df(smoother=False)

#     # @property
#     # def dfs(self) -> pd.DataFrame:
#     #     """History of the filter in the form of a pandas dataframe"""
#     #     return self.get_df(smoother=True)

#     # def plot_state_mean(
#     #     self,
#     #     state,
#     #     *args,
#     #     ax: plt.Axes | None = None,
#     #     smoother: bool = False,
#     #     **kwargs
#     # ) -> None:
#     #     """Plot the time history of state estimates"""
#     #     ax = plt.gca() if ax is None else ax
#     #     xvec = self.get_mean(state, smoother=smoother)
#     #     cb = ax.plot(self.tlist, xvec, *args, **kwargs)
#     #     state_idx = self.get_state_index(state)
#     #     ax.set_ylabel(f'{self.state_names[state_idx]}')
#     #     ax.set_xlim([self.tlist[0], self.tlist[-1]])
#     #     ax.set_xlabel('Time elapsed (Seconds)')
#     #     return cb

#     # def plot_state_cbound(
#     #     self,
#     #     state,
#     #     *args,
#     #     ax: plt.Axes | None = None,
#     #     smoother: bool = False,
#     #     cb_fac: float = 3.,
#     #     **kwargs
#     # ) -> None:
#     #     """Plot the time history of state estimates"""
#     #     ax = plt.gca() if ax is None else ax
#     #     xvec = self.get_mean(state, smoother=smoother)
#     #     xvec_var = self.get_cov(state, smoother=smoother)
#     #     cb_width = np.ravel(cb_fac * xvec_var**0.5)
#     #     cb = ax.fill_between(self.tlist, xvec - cb_width, xvec +
#     #                          cb_width, *args, **kwargs)
#     #     state_idx = self.get_state_index(state)
#     #     ax.set_ylabel(f'{self.state_names[state_idx]}')
#     #     ax.set_xlim([self.tlist[0], self.tlist[-1]])
#     #     ax.set_xlabel('Time elapsed (Seconds)')
#     #     return cb

#     # def plot_trajectory_cbound(
#     #     self,
#     #     x_indx: int,
#     #     y_indx: int,
#     #     ax: plt.Axes | None = None,
#     #     smoother: bool = False,
#     #     cb_fac: float = 3.,
#     #     **kwargs
#     # ) -> None:
#     #     """Plot the trajectory"""
#     #     ax = plt.gca() if ax is None else ax
#     #     xlocs = self.get_mean(x_indx, smoother=smoother)
#     #     ylocs = self.get_mean(y_indx, smoother=smoother)
#     #     xy_vars = self.get_cov([x_indx, y_indx], smoother=smoother)
#     #     for xloc, yloc, icov in zip(xlocs, ylocs, xy_vars):
#     #         width, height, angle = get_covariance_ellipse(icov, cb_fac)
#     #         ellip = Ellipse(xy=[xloc, yloc], width=width, height=height,
#     #                         angle=angle, **kwargs)
#     #         ax.add_artist(ellip)

#     # def plot_trajectory_mean(
#     #     self,
#     #     x_indx: int,
#     #     y_indx: int,
#     #     *args,
#     #     ax: plt.Axes | None = None,
#     #     smoother: bool = False,
#     #     **kwargs
#     # ) -> None:
#     #     """Plot the trajectory"""
#     #     ax = plt.gca() if ax is None else ax
#     #     xlocs = self.get_mean(x_indx, smoother=smoother)
#     #     ylocs = self.get_mean(y_indx, smoother=smoother)
#     #     cb = ax.plot(xlocs, ylocs, *args, **kwargs)
#     #     return cb

#     # @property
#     # def metrics(self) -> dict[str, float]:
#     #     """Return the summary metrics"""
#     #     idf = self.df
#     #     metric_names = list(self.cur_metrics.keys())
#     #     return idf.loc[~idf[metric_names[0]].isna(), metric_names].sum()

#     # @property
#     # def smetrics(self) -> dict[str, float]:
#     #     """Return the summary metrics"""
#     #     idf = self.dfs
#     #     metric_names = list(self.cur_smetrics.keys())
#     #     return idf.loc[~idf[metric_names[0]].isna(), metric_names].sum()

#     # def plot_metric(
#     #     self,
#     #     metric: str = 'NIS',
#     #     smoother=False,
#     #     ax: plt.Axes | None = None,
#     #     **kwargs
#     # ) -> None:
#     #     """Plot the time history of a performance metric"""
#     #     ax = plt.gca() if ax is None else ax
#     #     mnames = list(self.cur_metrics.keys())
#     #     if metric not in mnames:
#     #         self.raiseit(f'Invalid metric {metric}, choose from {mnames}')
#     #     idf = self.dfs if smoother else self.df
#     #     idfshort = idf[~idf[metric].isna()]
#     #     tvec = idfshort[self.time_colname].values
#     #     xvec = idfshort[metric].values
#     #     ax.plot(tvec, xvec, **kwargs)
#     #     ax.set_ylabel(metric)
#     #     ax.set_xlim([tvec[0], tvec[-1]])
#     #     ax.set_xlabel('Time elapsed (Seconds)')

#    # def compute_metrics(
#     #     self,
#     #     xres: ndarray | None = None,
#     #     xprec: ndarray | None = None,
#     #     yres: ndarray | None = None,
#     #     yprec: ndarray | None = None
#     # ):
#     #     """compute filter performance metrics"""
#     #     idict = {
#     #         'MetricXresNorm': np.nan,
#     #         'MetricYresNorm': np.nan,
#     #         'MetricNIS': np.nan,
#     #         'MetricNEES': np.nan,
#     #         'MetricLogLik': np.nan
#     #     }
#     #     if (yres is not None) and (yprec is not None):
#     #         idict['MetricYresNorm'] = np.linalg.norm(yres)
#     #         idict['MetricNIS'] = np.linalg.multi_dot([yres.T, yprec, yres])
#     #         prec_det = np.linalg.det(yprec)
#     #         idict['MetricLogLik'] = -0.5 * (self.ny * np.log(2. * np.pi) -
#     #                                         np.log(prec_det) + idict['MetricNIS'])
#     #     if (xres is not None) and (xprec is not None):
#     #         idict['MetricXresNorm'] = np.linalg.norm(xres)
#     #         idict['MetricNEES'] = np.linalg.multi_dot([xres.T, xprec, xres])
#     #     return idict

# # def initiate_state_by_dict(
# #     self,
# #     dict_of_mean_std: dict[str, tuple[float, float]]
# # ) -> None:
# #     """Initiate the time, state mean and covariance matrix"""
# #     self._m = np.zeros((self.nx,))
# #     self._P = np.eye(self.nx)
# #     for i, iname in enumerate(self.state_names):
# #         if iname in dict_of_mean_std.keys():
# #             self._m[i] = dict_of_mean_std[iname][0]
# #             self._P[i, i] = dict_of_mean_std[iname][1]**2


# class KalmanFilterBase(ABC):
#     """ Base class for implementing various versions of Kalman Filters"""

#     def __init__(
#         self,
#         nx: int,
#         ny: int,
#         dt: float,
#         object_id: str | int = 0
#     ):
#         # basic info
#         self.name: str = 'KFBase'
#         self.id: str = object_id
#         self.id_col = 'ObjectID'
#         self.dt: float = dt
#         self.dt_tol: float = 0.01
#         self.epsilon: float = 1e-20
#         self._nx: int = self.int_setter(nx)  # dimension of state vector
#         self._ny: int = self.int_setter(ny)  # dimension of observation vector
#         self._m: ndarray | None = None  # state mean vector
#         self._P: ndarray | None = None  # state covariance matrix
#         self._truth: ndarray | None = None
#         self._obs: ndarray | None = None
#         self.pars: Dict[str, float] = {}

#         # dynamics model
#         self.func_f: Callable | None = None  # state transition function
#         self._F: ndarray | None = None  # state transition matrix
#         self.func_Q: Callable | None = None
#         self._Q: ndarray | None = None  # process error cov mat
#         self._G: ndarray = np.eye(self.nx)  # Error jacobian matrix
#         self._qbar: ndarray = np.zeros((self.nx,))  # process error cov mat

#         # observation model
#         self.func_h: Callable | None = None  # observation-state function
#         self._H: ndarray | None = None  # observation-state matrix
#         self.func_R: Callable | None = None
#         self._R: ndarray | None = None  # obs error covariance matrix
#         self._J: ndarray = np.eye(self.ny)  # obs error jacobian matrix
#         self._rbar: ndarray = np.zeros((self.ny,))  # obs error mean

#         # add/subtract functions
#         self.x_add: Callable = np.add
#         self.x_subtract: Callable = np.subtract
#         self.y_subtract: Callable = np.subtract
#         self.x_mean_fn: Callable | None = None
#         self.y_mean_fn: Callable | None = None

#         # Kalman filtering parameters
#         self._nees: float = np.nan  # normalized estimation error squared
#         self._nis: float = np.nan  # normalized innovation squared
#         self._loglik: float = np.nan  # log lik of observations
#         self._time_elapsed: float = 0.  # time elapsed, default starts at 0.
#         self._last_update_at: float = 0.  # last update at this time
#         self._state_names: Sequence[str] = [f'x_{i}' for i in range(self._nx)]

#         # saving history of filter and smoother
#         varnames = ['TimeElapsed', 'Observation', 'Truth', 'ObservationCov',
#                     'FilterMean', 'FilterCov']
#         self.filter_metrics = ['FilterNEES', 'FilterNIS', 'FilterLogLik']
#         self.smoother_metrics = ['SmootherNEES',
#                                  'SmootherNIS', 'SmootherLogLik']
#         self._history = {}
#         for varname in varnames + self.filter_metrics:
#             self._history[varname] = []

#     def initiate_state(
#         self,
#         t0: float,
#         m0: ndarray,
#         P0: ndarray | None = None
#     ) -> None:
#         """Initiate the time, state mean and covariance matrix"""
#         for v in self._history.values():
#             v.clear()
#         self._time_elapsed = t0
#         self._m = self.vec_setter(m0, self.nx)
#         if P0 is not None:
#             self._P = self.mat_setter(P0, (self.nx, self.nx))
#         self._truth = None
#         self._obs = None
#         self._nees = np.nan
#         self._nis = np.nan
#         self._loglik = np.nan
#         self._last_update_at = t0
#         self._store_this_step()

#     def initiate_state_by_dict(
#         self,
#         t0: float,
#         dict_of_mean_std: Dict[str, Tuple[float, float]]
#     ) -> None:
#         """Initiate the time, state mean and covariance matrix"""
#         for v in self._history.values():
#             v.clear()
#         self._time_elapsed = t0
#         self._m = np.zeros((self.nx,))
#         self._P = np.eye(self.nx)
#         for i, iname in enumerate(self.state_names):
#             if iname in dict_of_mean_std.keys():
#                 self._m[i] = dict_of_mean_std[iname][0]
#                 self._P[i, i] = dict_of_mean_std[iname][1]**2
#         self._truth = None
#         self._obs = None
#         self._nees = np.nan
#         self._nis = np.nan
#         self._loglik = np.nan
#         self._store_this_step()

#     def forecast_upto(
#         self,
#         upto_time: float
#     ) -> None:
#         """Kalman filter forecast step upto some time in future"""
#         time_diff = upto_time - self.time_elapsed
#         if abs(time_diff) > self.dt_tol:
#             if time_diff < 0.:
#                 istr = f'{self.time_elapsed}->{upto_time}'
#                 self.raiseit(f'Forecasting backward in time:{istr}!')
#             nsteps = int(np.round(time_diff / self.dt))
#             for _ in range(nsteps):
#                 self.forecast()

#     def filter(
#         self,
#         list_of_time: Sequence[float],
#         list_of_obs: Sequence[ndarray],
#         list_of_R: Sequence[ndarray] | None = None,
#         list_of_truth: Sequence[ndarray] | None = None
#     ) -> None:
#         """Run filtering assuming F, H, Q matrices are time invariant"""

#         # check if inputs are valid
#         nsteps = len(list_of_time)
#         if len(list_of_obs) != nsteps:
#             print('Length mismatch!')
#             self.raiseit(f'time:{nsteps} vs observations: {len(list_of_obs)}')
#         if list_of_R is not None:
#             if len(list_of_R) != nsteps:
#                 self.raiseit('Length mismatch: time vs list of R matrices')
#         if list_of_truth is not None:
#             if len(list_of_truth) != nsteps:
#                 self.raiseit('Size mismatch: time vs list of state truths')
#         if len(self._history['TimeElapsed']) > 1:
#             self.raiseit('Need to initiate state!')

#         # run the forward filter
#         k = 0
#         while k < len(list_of_time):
#             if self._time_elapsed - list_of_time[k] > self.dt_tol:
#                 self.raiseit(f'Skipping observation, lower the dt={self.dt}!')
#             if abs(self._time_elapsed - list_of_time[k]) < self.dt_tol:
#                 self.obs = list_of_obs[k]
#                 self.R = self.R if list_of_R is None else list_of_R[k]
#                 self.truth = None if list_of_truth is None else list_of_truth[k]
#                 self.update()
#                 k += 1
#             else:
#                 self.forecast()
#                 if np.any(np.linalg.eigvals(self.P) < 0):
#                     print('Exiting because covariance is not pos def!')
#                     break

#     def smoother(self):
#         """Run smoothing assuming model/measurement eq are time invariant"""
#         nsteps = len(self.history['TimeElapsed'])
#         if nsteps < 1:
#             self.raiseit('No state history found, run filter first!')
#         cnames = ['SmootherMean', 'SmootherCov'] + self.smoother_metrics
#         for cname in cnames:
#             self._history[cname] = []
#         for i in reversed(range(nsteps)):
#             self._time_elapsed = self.history['TimeElapsed'][i]
#             self._obs = self.history['Observation'][i]
#             self._truth = self.history['Truth'][i]
#             self._m = deepcopy(self.history['FilterMean'][i])
#             self._P = deepcopy(self.history['FilterCov'][i])
#             self._R = self.history['ObservationCov'][i]
#             if i != nsteps - 1:
#                 smean_next = self.history['SmootherMean'][-1]
#                 scov_next = self.history['SmootherCov'][-1]
#                 self._backward_filter(smean_next, scov_next)
#             self._history['SmootherMean'].append(deepcopy(self.m))
#             self._history['SmootherCov'].append(deepcopy(self.P))
#             self._history[self.smoother_metrics[0]].append(self.nees)
#             self._history[self.smoother_metrics[1]].append(self.nis)
#             self._history[self.smoother_metrics[2]].append(self.loglik)
#         for k, v in self._history.items():
#             if k.startswith('Smoother'):
#                 v.reverse()

#     def _store_this_step(self, update: bool = False) -> None:
#         """Store this forecast/update step"""
#         self._time_elapsed = np.around(self.time_elapsed, 3)
#         if update:
#             for k, v in self._history.items():
#                 if not k.startswith('Smoother'):
#                     del v[-1]
#             self._last_update_at = self.time_elapsed
#         self._history['TimeElapsed'].append(self.time_elapsed)
#         self._history['Observation'].append(self.obs)
#         self._history['Truth'].append(self.truth)
#         self._history['ObservationCov'].append(self.R)
#         self._history['FilterMean'].append(deepcopy(self.m))
#         self._history['FilterCov'].append(deepcopy(self.P))
#         self._history[self.filter_metrics[0]].append(self.nees)
#         self._history[self.filter_metrics[1]].append(self.nis)
#         self._history[self.filter_metrics[2]].append(self.loglik)

#     def _compute_metrics(self, xres, xprec, yres, yprec):
#         """compute performance metrics"""
#         self._nis = np.linalg.multi_dot([yres.T, yprec, yres])
#         prec_det = np.linalg.det(yprec)
#         # if prec_det > 0.:
#         self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) -
#                                np.log(prec_det) + self.nis)
#         # else:
#         #    self._loglik = np.nan
#         if self.truth is not None:
#             xres = self.m - self.truth
#         self._nees = np.linalg.multi_dot([xres.T, xprec, xres])

#     def get_loglik_of_obs(
#         self,
#         y_obs: ndarray,
#         ignore_obs_inds: List[int] | None = None
#     ) -> None:
#         """
#         Compute log-likelihood of this observation
#         """
#         y_pred = self.H @ self.m
#         _residual = y_obs - y_pred
#         _this_smat = self.H @ self.P @ self.H.T + self.R
#         if ignore_obs_inds is not None:
#             for idx in ignore_obs_inds:
#                 _residual[idx] = 0.
#         _s_inv = np.linalg.pinv(_this_smat, hermitian=True)
#         # _s_inv = np.linalg.inv(_this_smat)
#         # print(_s_inv.diagonal())
#         this_loglik = self.ny * np.log(2. * np.pi)
#         this_loglik += np.log(np.linalg.det(_this_smat))
#         this_loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
#         this_loglik = np.linalg.multi_dot([_residual.T, _s_inv, _residual])
#         # this_loglik = np.dot(_residual, np.dot(_s_inv, _residual))
#         # this_loglik = np.dot(_residual, _residual)
#         this_loglik *= -0.5
#         return this_loglik

# ### abstract methods to be implemented by children of this base class ###

#     @ abstractmethod
#     def validate(self) -> None:
#         """Check if relevant matrices and/or functions have been initiated"""
#         if (self.P is None) or (self.m is None):
#             self.raiseit('Need to initiate state, use initiate_state()')
#         if self.R is None:
#             self.raiseit('Need to initiate R matrix!')
#         if self.Q is None:
#             self.raiseit('Need to initiate Q matrix!')

#     @ abstractmethod
#     def forecast(self) -> None:
#         """Forecast step"""
#         self._time_elapsed += np.around(self.dt, 3)
#         self._truth = None
#         self._obs = None
#         self._nis = np.nan
#         self._nees = np.nan
#         self._loglik = np.nan

#     @ abstractmethod
#     def update(self) -> None:
#         """Update step"""

#     @ abstractmethod
#     def _backward_filter(self, smean_next, scov_next) -> None:
#         """Backward filter"""


# ### accessing filter/smoother results###

#     def get_mean(
#         self,
#         this_state: str | int,
#         smoother: bool = False
#     ) -> ndarray:
#         """Get state mean time series of ith state"""
#         cname = 'SmootherMean' if smoother else 'FilterMean'
#         if isinstance(this_state, int):
#             idx = this_state
#             self.check_state_index(idx)
#         elif isinstance(this_state, str):
#             assert this_state in self.state_names, f'{this_state} invalid!'
#             idx = self.state_names.index(this_state)
#         return np.stack(self.df[cname].values)[:, idx]

#     def get_time_elapsed(self) -> ndarray:
#         """Get time elapsed"""
#         return self.df['TimeElapsed'].values

#     def get_cov(
#         self,
#         this_state: str | int | Tuple[int, int] | Tuple[str, str],
#         smoother: bool = False
#     ) -> ndarray:
#         """Get state covariance matrix for the desired indices"""
#         cname = 'SmootherCov' if smoother else 'FilterCov'
#         if isinstance(this_state, int):
#             idx_pair = [this_state, this_state]
#         elif isinstance(this_state, str):
#             assert this_state in self.state_names, f'{this_state} invalid!'
#             idx = self.state_names.index(this_state)
#             idx_pair = [idx, idx]
#         elif isinstance(this_state, Sequence):
#             assert len(this_state) == 2, 'Need a pair of state idx/names!'
#             idx_pair = [0, 0]
#             for i, idx in enumerate(this_state):
#                 if isinstance(idx, int):
#                     self.check_state_index(idx)
#                     idx_pair[i] = idx
#                 elif isinstance(idx, str):
#                     self.check_state_name(idx)
#                     idx_pair[i] = self.state_names.index(idx)
#                 else:
#                     self.raiseit('Need either int or str!')
#         else:
#             self.raiseit('Need either int or str or (int,int) or (str,str)')
#         # indices = np.ix_(np.arange(self.df.shape[0]), idx_pair[0], idx_pair[1])
#         return np.stack(self.df[cname].values)[:, idx_pair[0], idx_pair[1]]

#     # def get_cov(
#     #     self,
#     #     row_inds: int,
#     #     col_inds: int | None = None,
#     #     smoother: bool = False
#     # ) -> ndarray:
#     #     """Get state covariance matrix for the desired indices"""
#     #     cname = 'SmootherCov' if smoother else 'FilterCov'
#     #     row_inds = row_inds if isinstance(row_inds, list) else [row_inds]
#     #     col_inds = col_inds if col_inds is not None else row_inds
#     #     col_inds = col_inds if isinstance(col_inds, list) else [col_inds]
#     #     indices = np.ix_(np.arange(self.df.shape[0]), row_inds, col_inds)
#     #     return np.stack(self.df[cname].values)[indices]

#     @ property
#     def metrics(self) -> dict:
#         """Get the summary performance metrics"""
#         out_metrics = {}
#         for cname in self.filter_metrics + self.smoother_metrics:
#             if cname in list(self.df.columns):
#                 out_metrics[cname] = np.around(
#                     self.df[cname].dropna().sum(), 3)
#         return out_metrics


# ### Plotting related ###

#     def plot_state_mean(
#         self,
#         this_state,
#         *args,
#         ax: plt.Axes | None = None,
#         smoother: bool = False,
#         **kwargs
#     ) -> None:
#         """Plot the time history of state estimates"""
#         ax = plt.gca() if ax is None else ax
#         tvec = self.get_time_elapsed()
#         xvec = self.get_mean(this_state, smoother=smoother)
#         cb = ax.plot(tvec, xvec, *args, **kwargs)
#         # ax.set_ylabel(f'{self.state_names[state_idx]}')
#         ax.set_xlim([tvec[0], tvec[-1]])
#         # ax.set_xlabel('Time elapsed (Seconds)')
#         return cb

#     def plot_state_cbound(
#         self,
#         this_state,
#         *args,
#         ax: plt.Axes | None = None,
#         smoother: bool = False,
#         cb_fac: float = 3.,
#         **kwargs
#     ) -> None:
#         """Plot the time history of state estimates"""
#         ax = plt.gca() if ax is None else ax
#         tvec = self.get_time_elapsed()
#         xvec = self.get_mean(this_state, smoother=smoother)
#         xvec_var = self.get_cov(this_state, smoother=smoother)
#         cb_width = np.ravel(cb_fac * xvec_var**0.5)
#         cb = ax.fill_between(tvec, xvec - cb_width, xvec +
#                              cb_width, *args, **kwargs)
#         # ax.set_ylabel(f'{self.state_names[state_index]}')
#         ax.set_xlim([tvec[0], tvec[-1]])
#         # ax.set_xlabel('Time elapsed (Seconds)')
#         return cb

#     # def plot(
#     #     self,
#     #     smoother: bool = False,
#     #     fig_size=(6, 6)
#     # ) -> None:
#     #     """Plot the time history of state estimates"""
#     #     self.nx
#     #     fig, ax = plt.subplots(, figsize=fig_size)
#     #     ax = plt.gca() if ax is None else ax
#     #     tvec = self.get_time_elapsed()
#     #     xvec = self.get_mean(this_state, smoother=smoother)
#     #     cb = ax.plot(tvec, xvec, *args, **kwargs)
#     #     # ax.set_ylabel(f'{self.state_names[state_idx]}')
#     #     ax.set_xlim([tvec[0], tvec[-1]])
#     #     # ax.set_xlabel('Time elapsed (Seconds)')
#     #     return cb

#     def plot_trajectory_cbound(
#         self,
#         x_indx: int,
#         y_indx: int,
#         ax: plt.Axes | None = None,
#         smoother: bool = False,
#         cb_fac: float = 3.,
#         **kwargs
#     ) -> None:
#         """Plot the trajectory"""
#         self.check_state_index(x_indx)
#         self.check_state_index(y_indx)
#         ax = plt.gca() if ax is None else ax
#         xlocs = self.get_mean(x_indx, smoother=smoother)
#         ylocs = self.get_mean(y_indx, smoother=smoother)
#         xy_vars = self.get_cov([x_indx, y_indx], smoother=smoother)
#         for xloc, yloc, icov in zip(xlocs, ylocs, xy_vars):
#             width, height, angle = get_covariance_ellipse(icov, cb_fac)
#             ellip = Ellipse(xy=[xloc, yloc], width=width, height=height,
#                             angle=angle, **kwargs)
#             ax.add_artist(ellip)

#     def plot_trajectory_mean(
#         self,
#         x_indx: int,
#         y_indx: int,
#         *args,
#         ax: plt.Axes | None = None,
#         smoother: bool = False,
#         **kwargs
#     ) -> None:
#         """Plot the trajectory"""
#         self.check_state_index(x_indx)
#         self.check_state_index(y_indx)
#         ax = plt.gca() if ax is None else ax
#         xlocs = self.get_mean(x_indx, smoother=smoother)
#         ylocs = self.get_mean(y_indx, smoother=smoother)
#         cb = ax.plot(xlocs, ylocs, *args, **kwargs)
#         return cb

#     def plot_metric(
#         self,
#         ax: plt.Axes | None = None,
#         metric: str = 'FilterNIS',
#         **kwargs
#     ) -> None:
#         """Plot the time history of a performance metric"""
#         ax = plt.gca() if ax is None else ax
#         if metric not in self.df.columns:
#             self.raiseit(f'Invalid metric name {metric}')
#         dfshort = self.df[~self.df[metric].isna()]
#         tvec = dfshort['TimeElapsed'].values
#         xvec = dfshort[metric].values
#         ax.plot(tvec, xvec, **kwargs)
#         ax.set_ylabel(metric)
#         ax.set_xlim([tvec[0], tvec[-1]])
#         ax.set_xlabel('Time elapsed (Seconds)')

# ### Matrix/vector handling###

#     def __str__(self) -> SyntaxWarning:
#         """Print output"""
#         out_str = f'----{self.name}-{self.id}----\n'
#         out_str += f'State labels     : ' + ','.join(self.state_names) + '\n'
#         out_str += f'State dimension  : {self._nx}\n'
#         out_str += f'Observation dim  : {self._ny}\n'
#         out_str += f'Time interval dt : {self.dt}\n'
#         out_str += f'Tolerance in dt  : {self.dt_tol}\n'
#         return out_str

#     def check_state_index(self, indx: int):
#         """Check the validity of the state index"""
#         if indx > self.nx:
#             self.raiseit(f'Invalid {indx}, choose state_index < {self.nx}')

#     def check_state_name(self, iname: str):
#         """Check the validity of the state name"""
#         if iname not in self.state_names:
#             self.raiseit(
#                 f'Invalid state {iname}, choose from {self.state_names}')

#     def mat_setter(self, in_mat, to_shape=None) -> ndarray:
#         """Returns a valid numpy array2d while checking for its shape"""
#         in_mat = np.atleast_2d(np.asarray_chkfinite(in_mat, dtype=float))
#         if in_mat.ndim != 2:
#             self.raiseit(f'Need 2d array, input dim: {in_mat.ndim}')
#         if to_shape is not None:
#             if in_mat.shape != to_shape:
#                 print('Shape mismatch!')
#                 self.raiseit(f'Required: {to_shape}, Input: {in_mat.shape}')
#         return in_mat

#     def vec_setter(self, in_vec, to_size=None) -> ndarray:
#         """Returns a valid numpy array1d while checking for its shape"""
#         in_vec = np.atleast_1d(np.asarray_chkfinite(in_vec, dtype=float))
#         in_vec = in_vec.flatten()
#         if to_size is not None:
#             if in_vec.size != to_size:
#                 print('Size mismatch!')
#                 self.raiseit(f'Required: {to_size}, Input: {in_vec.size}')
#         return in_vec

#     def float_setter(self, in_val) -> float:
#         """Return a valid scalar"""
#         in_val = np.asarray_chkfinite(in_val, dtype=float)
#         if in_val.size != 1:
#             self.raiseit(f'Need scalar!, change input:{in_val} to scalar')
#         return float(in_val.item())

#     def int_setter(self, in_val) -> int:
#         """Return a valid scalar"""
#         in_val = np.asarray_chkfinite(in_val, dtype=int)
#         if in_val.size != 1:
#             self.raiseit(f'Need scalar!, change input:{in_val} to scalar')
#         return int(in_val.item())

#     def raiseit(self, outstr: str = "") -> None:
#         """Raise exception with the out string"""
#         raise ValueError(f'{self.name}: {outstr}')


# ### Getter for private class variables at last update/forecast###


#     @ property
#     def nx(self) -> int:
#         """Dimension of state space"""
#         return self._nx

#     @ property
#     def ny(self) -> int:
#         """Dimension of observation space"""
#         return self._ny

#     @ property
#     def m(self) -> ndarray:
#         """State mean"""
#         return self._m

#     @ property
#     def P(self) -> ndarray:
#         """State covariance matrix"""
#         return self._P

#     @property
#     def lifespan(self):
#         """Returns the time duration of existence till the last update"""
#         return self.last_update_at - self.get_time_elapsed()[0]

#     @ property
#     def time_elapsed(self) -> float:
#         """Time elapsed so far """
#         return self._time_elapsed

#     @ property
#     def last_update_at(self) -> float:
#         """Time elapsed so far """
#         return self._last_update_at

#     @ property
#     def nis(self) -> float:
#         """Normalized innovation squared at the last update"""
#         return self._nis

#     @ property
#     def nees(self) -> float:
#         """Normalized estimation error squared at the last update"""
#         return self._nees

#     @ property
#     def loglik(self) -> float:
#         """Normalized estimation error squared at the last update"""
#         return self._loglik

#     @ property
#     def history(self) -> pd.DataFrame:
#         """History of the filter in the form of a dictionary"""
#         return self._history

#     @ property
#     def df(self) -> pd.DataFrame:
#         """History of the filter/smoother in the form of a pandas dataframe"""
#         return pd.DataFrame(self.history)

#     @ property
#     def df_filter(self) -> pd.DataFrame:
#         """History of the filter in the form of a pandas dataframe"""
#         smoother = False
#         dff = self.df.loc[:, ['TimeElapsed']].copy()
#         for iname in self.state_names:
#             dff[iname] = self.get_mean(iname, smoother=smoother)
#             dff[iname + '_var'] = self.get_cov(iname, smoother=smoother)
#         dff['Observation'] = self.df['Observation']
#         dff['Training'] = True
#         dff.loc[self.df['Observation'].isna(), 'Training'] = False
#         dff[self.id_col] = self.id
#         for cname in self.filter_metrics:
#             dff[cname] = self.df[cname]
#         # dff[dff.select_dtypes(np.float64).columns] = dff.select_dtypes(
#         #     np.float64).astype(np.float32)
#         return dff

#     @ property
#     def df_smoother(self) -> pd.DataFrame:
#         """History of the filter in the form of a pandas dataframe"""
#         if 'SmootherMean' not in self.df.columns:
#             self.raiseit('No smoother history found. Run smoother first!')
#         smoother = True
#         dff = self.df.loc[:, ['TimeElapsed']].copy()
#         for iname in self.state_names:
#             dff[iname] = self.get_mean(iname, smoother=smoother)
#             dff[iname + '_var'] = self.get_cov(iname, smoother=smoother)
#         dff['Training'] = True
#         dff.loc[self.df['Observation'].isna(), 'Training'] = False
#         dff[self.id_col] = self.id
#         for cname in self.smoother_metrics:
#             dff[cname] = self.df[cname]
#         # dff[dff.select_dtypes(np.float64).columns] = dff.select_dtypes(
#         #     np.float64).astype(np.float32)
#         # dff[dff.select_dtypes(np.int64).columns] = dff.select_dtypes(
#         #     np.int64).astype(np.int32)
#         return dff

# ### Getter/Setter for Truth and observations###

#     @ property
#     def truth(self) -> ndarray:
#         """State truth"""
#         return self._truth

#     @ truth.setter
#     def truth(self, in_mat: ndarray | None) -> None:
#         """Setter for state truth"""
#         self._truth = self.vec_setter(
#             in_mat, self.nx) if in_mat is not None else None

#     @ property
#     def obs(self) -> ndarray:
#         """Observation vector"""
#         return self._obs

#     @ obs.setter
#     def obs(self, in_mat: ndarray | None) -> None:
#         """Setter for observation vector"""
#         self._obs = self.vec_setter(
#             in_mat, self.ny) if in_mat is not None else None

#     @ property
#     def state_names(self) -> list:
#         """Labels"""
#         return self._state_names

#     @ state_names.setter
#     def state_names(self, in_val: Sequence[str]) -> None:
#         """Setter for labels"""
#         if len(in_val) != self.nx:
#             self.raiseit(f'Number of labels should be {self.nx}')
#         self._state_names = [ix for ix in in_val]


# ### Getter/Setter for matrices of dynamics model###

#     @ property
#     def F(self) -> ndarray:
#         """State transition matrix"""
#         return self._F

#     @ F.setter
#     def F(self, in_mat: ndarray) -> None:
#         """Setter for state transition matrix"""
#         self._F = self.mat_setter(in_mat, (self.nx, self.nx))

#     @ property
#     def G(self) -> ndarray:
#         """Model error Jacobian matrix"""
#         return self._G

#     @ G.setter
#     def G(self, in_mat: ndarray) -> None:
#         """Setter for state transition matrix"""
#         self._G = self.mat_setter(in_mat, (self.nx, self.nx))

#     @ property
#     def Q(self) -> ndarray:
#         """Process error covariance matrix"""
#         return self._Q

#     @ Q.setter
#     def Q(self, in_mat: ndarray) -> None:
#         """Setter for process error covariance matrix"""
#         self._Q = self.mat_setter(in_mat, (self.nx, self.nx))

#     @ property
#     def qbar(self) -> ndarray:
#         """Process error mean"""
#         return self._qbar

#     @ qbar.setter
#     def qbar(self, in_mat: ndarray) -> None:
#         """Setter for process error mean"""
#         self._qbar = self.vec_setter(in_mat, self.nx)

# ### Getter/Setter for matrices of observation model###

#     @ property
#     def H(self) -> ndarray:
#         """observation matrix"""
#         return self._H

#     @ H.setter
#     def H(self, in_mat: ndarray) -> None:
#         """Setter for observation-state relation"""
#         self._H = self.mat_setter(in_mat, (self.ny, self.nx))

#     @ property
#     def J(self) -> ndarray:
#         """observation error jacobian matrix"""
#         return self._J

#     @ J.setter
#     def J(self, in_mat: ndarray) -> None:
#         """Setter for observation-state relation"""
#         self._J = self.mat_setter(in_mat, (self.ny, self.ny))

#     @ property
#     def R(self) -> ndarray:
#         """observation error covariance matrix"""
#         return self._R

#     @ R.setter
#     def R(self, in_mat: ndarray) -> None:
#         """Setter for observation error covariance matrix"""
#         self._R = self.mat_setter(in_mat, (self.ny, self.ny))

#     @ property
#     def rbar(self) -> ndarray:
#         """Observation error mean"""
#         return self._rbar

#     @ rbar.setter
#     def rbar(self, in_mat: ndarray) -> None:
#         """Setter for obs error mean"""
#         self._rbar = self.vec_setter(in_mat, self.ny)

#     @ staticmethod
#     def symmetrize(in_mat: ndarray) -> ndarray:
#         """Return a symmetrized version of NumPy array"""
#         # if np.any(np.isnan(in_mat)) or np.any(in_mat.diagonal() < 0.):
#         #     print('\np update went wrong!')
#         #     print(in_mat.diagonal())
#         return (in_mat + in_mat.T) / 2.


# class KalmanFilter:
#     # pylint: disable=too-many-instance-attributes
#     # pylint: disable=too-many-public-methods
#     # pylint: disable=invalid-name
#     """ Class for implementing Kalman Filter """

#     def __init__(self,
#                  dim_x: int,
#                  dim_y: int = 1,
#                  object_id: Union[str, int] = 0,
#                  dt_tol: float = 0.001
#                  ):
#         self._id = str(object_id)
#         self._dim_x: int = int(dim_x) if isinstance(dim_x, float) else dim_x
#         self._dim_y: int = int(dim_y) if isinstance(dim_x, float) else dim_y
#         self._m: ndarray = np.zeros(dim_x)  # state mean vector
#         self._P: ndarray = np.eye(dim_x)  # state covariance matrix
#         self._F: ndarray = np.eye(dim_x)  # state transition matrix
#         self._Q: ndarray = np.eye(dim_x)  # model error covariance matrix
#         self._H: ndarray = np.zeros((dim_y, dim_x))  # observation function
#         self._R: ndarray = np.eye(dim_y)  # obs error covariance matrix
#         self._K: ndarray = np.zeros((dim_y, dim_x))  # kalman gain matrix
#         self._S: ndarray = np.zeros((dim_y, dim_y))  # innovation matrix
#         self._loglik: float = 0.  # likelihood of data given predicted state
#         self._time_elapsed: float = 0.  # time elapsed, default starts at 0.
#         self._last_update_at: float = 0.  # last update at this time
#         self._dt_tolerance: float = dt_tol
#         self._time_history = []
#         self._filter_mean = []
#         self._filter_cov = []
#         self._filter_loglik = []
#         self._smoother_mean = []
#         self._smoother_cov = []
#         self._smoother_loglik = []
#         self._observations = []
#         self.labels = [f'x_{i}' for i in range(self._dim_x)]

#     def filter(
#         self,
#         dt_pred: float,
#         tobs: ndarray,
#         yobs: ndarray,
#         yobs_var: ndarray
#     ) -> None:
#         """
#         Run filtering assuming F, H, Q matrices are time invariant
#         """
#         k = 1
#         self._time_elapsed = tobs[0]
#         while k < tobs.size:
#             if self._time_elapsed - tobs[k] > self._dt_tolerance:
#                 raise Exception('Skipping an observation, lower the dt!')
#             if abs(self._time_elapsed - tobs[k]) < self._dt_tolerance:
#                 np.fill_diagonal(self._R, yobs_var[k, :])
#                 self.update(yobs[k, :])
#                 k += 1
#             else:
#                 self.forecast(dt_pred)

#     def forecast_upto(self, upto_time: float, time_dt: float) -> None:
#         """
#         Kalman filter forecast step upto some time in future
#         """
#         time_diff = upto_time - self.time_elapsed
#         if abs(time_diff) > self._dt_tolerance:
#             if time_diff < 0.:
#                 raise Exception('KF: Forecasting back in time!')
#             nsteps = int(np.round(time_diff / time_dt))
#             # print(f'going from {self.time_elapsed} to {upto_time} in {nsteps}')
#             for _ in range(nsteps):
#                 self.forecast(time_dt)

#     def forecast(self, time_dt: Optional[float] = None) -> None:
#         """
#         Kalman filter forecast step
#         """
#         self._m = np.dot(self._F, self._m)
#         self._P = np.matmul(self._F, np.matmul(self._P, self._F.T)) + self._Q
#         self._time_elapsed += time_dt if time_dt is not None else 1.
#         self._loglik = np.nan
#         self._store_this_step()

#     def update(self, y_obs: ndarray) -> None:
#         """
#         Kalman filter update step
#         """
#         y_obs = self.get_valid_obs(y_obs)
#         y_pred = self._H @ self._m
#         _residual = y_obs - y_pred
#         self._S = self._H @ self._P @ self._H.T + self._R
#         _s_inv = np.linalg.pinv(self._S, hermitian=True)
#         self._loglik = self.get_loglik_of_obs(y_obs)
#         self._K = self._P @ self._H.T @ _s_inv
#         self._m = self._m + np.dot(self._K, _residual)
#         self._P = (np.eye(self._dim_x) - self._K @ self._H) @ self._P
#         self._store_this_step(obs=y_obs)

#     def initiate(self, in_time: float, in_m: ndarray, in_pmat: ndarray):
#         """
#         Initiate the state mean and covariance matrix
#         """
#         in_m = np.asarray(in_m) if not isinstance(in_m, ndarray) else in_m
#         in_pmat = np.asarray(in_pmat) if not isinstance(
#             in_pmat, ndarray) else in_pmat
#         assert in_m.ndim == 1, 'KF: Need 1D mean vector'
#         assert in_pmat.ndim == 2, 'KF: Need 2D covaince matrix'
#         assert in_m.size == self._dim_x, 'KF: Wrong size for mean vector!'
#         assert in_pmat.shape == (
#             self._dim_x, self._dim_x), 'KF: Wrong size for mean vector!'
#         self._m = in_m
#         self._P = in_pmat
#         self._time_elapsed = in_time
#         self._loglik = np.nan
#         self._store_this_step(first_update=True)

#     def get_loglik_of_obs(self, y_obs: ndarray) -> None:
#         """
#         Compute log-likelihood of this observation
#         """
#         y_obs = self.get_valid_obs(y_obs)
#         y_pred = self._H @ self._m
#         # print(y_pred)
#         _residual = y_obs - y_pred
#         _this_smat = self._H @ self._P @ self._H.T + self._R
#         _s_inv = np.linalg.pinv(_this_smat, hermitian=True)
#         # _s_inv = np.linalg.inv(_this_smat)
#         # print(_s_inv.diagonal())
#         this_loglik = self._dim_y * np.log(2. * np.pi)
#         this_loglik += np.log(np.linalg.det(_this_smat))
#         this_loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
#         this_loglik = np.linalg.multi_dot([_residual.T, _s_inv, _residual])
#         # this_loglik = np.dot(_residual, np.dot(_s_inv, _residual))
#         # this_loglik = np.dot(_residual, _residual)
#         this_loglik *= -0.5
#         return this_loglik

#     def compute_innovation_metrics(self, y_obs: ndarray) -> None:
#         """
#         Compute log-likelihood and innovation distance of this observation
#         """
#         y_obs = np.asarray(y_obs)
#         self._check_valid_obs(y_obs)
#         y_pred = self._hmat @ self._mvec
#         _innov = y_obs - y_pred
#         _smat = self._hmat @ self._pmat @ self._hmat.T + self._rmat
#         _smat_inv = np.linalg.pinv(_smat, hermitian=True)
#         # _smat_inv = np.linalg.inv(_smat)
#         self._nis = np.linalg.multi_dot([_innov.T, _smat_inv, _innov])
#         self._loglik = self._dim_obs * np.log(2. * np.pi)
#         self._loglik += np.log(np.linalg.det(_smat))
#         self._loglik += self._nis
#         self._loglik *= -0.5
#         # return loglik, norm_innov_dist, innov_dist

#     def filter(
#         self,
#         tobs: ndarray,
#         yobs: ndarray,
#         yobs_var: ndarray,
#         dt_tol: float = 0.001
#     ) -> None:
#         """
#         Run filtering assuming model/measurement equations are time invariant
#         """
#         k = 1
#         self._time_elapsed = tobs[0]
#         while k < tobs.size:
#             if self._time_elapsed - tobs[k] > dt_tol:
#                 self._raiseit('Skipping an observation, lower the dt!')
#             if abs(self._time_elapsed - tobs[k]) < dt_tol:
#                 np.fill_diagonal(self._R, yobs_var[k, :])
#                 self.update(yobs[k, :])
#                 k += 1
#             else:
#                 self.forecast(self.dt)

#     def smoother(self):
#         """
#         Run smoothing assuming model/measurement equations are time invariant
#         """
#         num_steps = np.asarray(self._filter_mean).shape[0]
#         self._smoother_mean = []
#         self._smoother_cov = []
#         self._smoother_loglik = []
#         for i in reversed(range(num_steps)):
#             if i == num_steps - 1:
#                 _smean = self._filter_mean[i]
#                 _scov = self._filter_cov[i]
#             else:
#                 _umean = self._filter_mean[i]
#                 _ucov = self._filter_cov[i]
#                 _fcov = self._fmat @ _ucov @ self._fmat.T + self._qmat
#                 _fcov_inv = np.linalg.pinv(_fcov)
#                 _gmat = _ucov @ self._fmat.T @ _fcov_inv
#                 _smean = _umean + _gmat @ (_smean - self._fmat @ _umean)
#                 _scov = _ucov + _gmat @ (_scov - _fcov) @ _gmat.T
#             self._smoother_mean.append(_smean)
#             self._smoother_cov.append(_scov)
#         self._smoother_mean.reverse()
#         self._smoother_cov.reverse()
#         self._compute_innovation_metrics_smoother()

#     def _compute_innovation_metrics_smoother(self):
#         """
#         Computes innovation metrics for the smoother
#         """
#         k = 1
#         for i, _floglik in enumerate(self._filter_nis):
#             if not np.isnan(_floglik):
#                 _scov_prev = self._smoother_cov[i - 1]
#                 _ypred = self._hmat @ self._fmat @ self._smoother_mean[i - 1]
#                 _innov = self._observations[k, :] - _ypred
#                 _smat = self._hmat @ _scov_prev @ self._hmat.T + self._rmat
#                 _smat_inv = np.linalg.pinv(_smat, hermitian=True)
#                 _snis = np.linalg.multi_dot([_innov.T, _smat_inv, _innov])
#                 _loglik = self._dim_obs * np.log(2. * np.pi)
#                 _loglik += np.log(np.linalg.det(_smat))
#                 _loglik += _snis
#                 _loglik *= -0.5
#                 k += 1
#             else:
#                 _loglik = np.nan
#                 _snis = np.nan
#             self._smoother_loglik.append(_loglik)
#             self._smoother_nis.append(_snis)

#     def get_valid_obs(self, y_obs: ndarray) -> ndarray:
#         """Get observations in a compatible form and also check validity"""
#         y_obs = np.asarray(y_obs)
#         y_obs = y_obs.flatten()
#         assert y_obs.size == self._H.shape[0], 'Incompatible obs with H matrix'
#         assert y_obs.size == self._R.shape[0], 'Incompatible obs with R matrix'
#         assert y_obs.size == self._R.shape[1], 'Incompatible obs with R matrix'
#         return y_obs

#     def smoother(self):
#         """ Run smoothing assuming F, H, Q matrices are time invariant """
#         num_steps = np.asarray(self._filter_mean).shape[0]
#         self._smoother_mean = []
#         self._smoother_cov = []
#         self._smoother_loglik = []
#         for i in reversed(range(num_steps)):
#             if i == num_steps - 1:
#                 _smean = self._filter_mean[i]
#                 _scov = self._filter_cov[i]
#             else:
#                 _umean = self._filter_mean[i]
#                 _ucov = self._filter_cov[i]
#                 _fcov = self._F @ _ucov @ self._F.T + self._Q
#                 _fcov_inv = np.linalg.pinv(_fcov)
#                 _gmat = _ucov @ self._F.T @ _fcov_inv
#                 _smean = _umean + _gmat @ (_smean - self._F @ _umean)
#                 _scov = _ucov + _gmat @ (_scov - _fcov) @ _gmat.T
#             self._smoother_mean.append(_smean)
#             self._smoother_cov.append(_scov)
#         self._smoother_mean.reverse()
#         self._smoother_cov.reverse()
#         self._compute_loglik_smoother()

#     def _compute_loglik_smoother(self):
#         """ Computes loglik of observations given smooothened estimates """
#         k = 1
#         for i, _floglik in enumerate(self._filter_loglik):
#             if not np.isnan(_floglik):
#                 _scov_prev = self._smoother_cov[i - 1]
#                 _ypred = self._H @ self._F @ self._smoother_mean[i - 1]
#                 _residual = self._observations[k, :] - _ypred
#                 _smat = self._H @ _scov_prev @ self._H.T + self._R
#                 _s_inv = np.linalg.pinv(_smat, hermitian=True)
#                 _loglik = self._dim_y * np.log(2. * np.pi)
#                 _loglik += np.log(np.linalg.det(_smat))
#                 _loglik += np.linalg.multi_dot([_residual.T,
#                                                 _s_inv, _residual])
#                 _loglik *= -0.5
#                 k += 1
#             else:
#                 _loglik = np.nan
#             self._smoother_loglik.append(_loglik)

#     def _store_this_step(self, obs: Optional[ndarray] = None,
#                          first_update=False):
#         if not first_update:
#             time_diff = self._time_history[-1] - self.time_elapsed
#             if abs(time_diff) < self._dt_tolerance:
#                 self._remove_last_entry()
#         else:
#             self._last_update_at = self._time_elapsed
#         self._filter_loglik.append(self._loglik)
#         self._time_history.append(self._time_elapsed)
#         self._filter_mean.append(self._m)
#         self._filter_cov.append(self._P)
#         if obs is not None:
#             self._observations.append(obs)
#             self._last_update_at = self._time_elapsed

#     def _remove_last_entry(self) -> None:
#         """ Remove last entry in the record keeping """
#         del self._time_history[-1]
#         del self._filter_mean[-1]
#         del self._filter_cov[-1]
#         del self._filter_loglik[-1]

#     def plot_loglik(
#         self,
#         time_window: Optional[Tuple[float, float]] = None
#     ):
#         """ plots the time history of log-likelihood of observations """
#         if len(self._time_history) < 1:
#             raise ValueError('KF: No state history found, run filter first!')
#         fig, ax = plt.subplots(figsize=(6, 2.5))
#         fdf = self.df
#         fdfshort = fdf[~fdf['loglik'].isna()]
#         ax.plot(fdfshort['time_elapsed'], fdfshort['loglik'], '.b')
#         ax.set_ylabel('Log-likelihood of obs')
#         ax.grid(True)
#         if time_window is None:
#             ax.set_xlim([fdfshort['time_elapsed'].iloc[0],
#                          fdfshort['time_elapsed'].iloc[-1]])
#         else:
#             ax.set_xlim(time_window)
#         ax.set_xlabel('Time elapsed (Seconds)')
#         fig.tight_layout()
#         return fig, ax

#     def plot_filtered_state(
#         self,
#         ax,
#         state_index: int,
#         lcolor='b',
#         time_window: Optional[Tuple[float, float]] = None,
#         cb_fac=3.
#     ):
#         """ plots the time history of state estimates """
#         assert state_index < self._dim_x, f'Choose state_index < {self._dim_x}'
#         m_name, p_name = self.get_label_for_this(state_index)
#         idf = self.df.copy()
#         t_history = idf['time_elapsed'].values / 60.
#         m_history = idf[m_name].values
#         cb_history = cb_fac * idf.loc[:, p_name].values**0.5
#         ax.plot(t_history, m_history, linestyle='-', color=lcolor,
#                 label='Kalman Filter')
#         ax.fill_between(t_history, m_history - cb_history,
#                         m_history + cb_history,
#                         fc=lcolor, ec='none', alpha=0.25)
#         ax.set_ylabel(f'{self.labels[state_index]}')
#         if time_window is None:
#             ax.set_xlim([t_history[0], t_history[-1]])
#         else:
#             ax.set_xlim(time_window)
#         ax.set_xlabel('Time elapsed (minutes)')

#     def plot_smoother_state(
#         self,
#         ax,
#         state_index: int,
#         lcolor='b',
#         time_window: Optional[Tuple[float, float]] = None,
#         cb_fac=3.
#     ):
#         """ plots the time history of state estimates """
#         assert state_index < self._dim_x, f'Choose state_index < {self._dim_x}'
#         m_name, p_name = self.get_label_for_this(state_index)
#         idf = self.dfs.copy()
#         if idf.shape != self.df.shape:
#             self.smoother()
#             idf = self.dfs.copy()
#         t_history = idf['time_elapsed'].values / 60.
#         m_history = idf[m_name].values
#         cb_history = cb_fac * idf.loc[:, p_name].values**0.5
#         ax.plot(t_history, m_history, linestyle='-', color=lcolor,
#                 label='Kalman Smoother')
#         ax.fill_between(t_history, m_history - cb_history,
#                         m_history + cb_history,
#                         fc=lcolor, ec='none', alpha=0.25)
#         ax.set_ylabel(f'{self.labels[state_index]}')
#         if time_window is None:
#             ax.set_xlim([t_history[0], t_history[-1]])
#         else:
#             ax.set_xlim(time_window)
#         ax.set_xlabel('Time elapsed (minutes)')

#     def plot_observations(
#         self,
#         ax,
#         obs_index: int,
#         lcolor='r',
#     ):
#         """ plots the time history of state estimates """
#         idf = self.df.copy()
#         iudf = idf[~idf['loglik'].isna()]
#         ax.plot(iudf['time_elapsed'] / 60., self.observations[:, obs_index],
#                 '*', color=lcolor, mec=None, alpha=0.5,
#                 markersize=5., label='Observations')

#     def get_label_for_this(self, idx: int):
#         """Get labels for this state index"""
#         return (self.labels[idx], f'var_{self.labels[idx]}')

#     @property
#     def df(self) -> pd.DataFrame:
#         """Dataframe containing entire history of the filtering"""
#         hdict = {'object_id': np.asarray([self._id] * len(self._time_history))}
#         hdict.update({'time_elapsed': np.asarray(self._time_history)})
#         hdict.update({'loglik': np.asarray(self._filter_loglik)})
#         for i in range(self._dim_x):
#             m_name, p_name = self.get_label_for_this(i)
#             hdict.update({m_name: np.asarray(self._filter_mean)[:, i]})
#             hdict.update({p_name: np.asarray(self._filter_cov)[:, i, i]})
#         return pd.DataFrame(hdict)

#     @property
#     def dfs(self) -> pd.DataFrame:
#         """Dataframe containing entire history of the smoothing"""
#         hdict = {'time_elapsed': np.asarray(self._time_history)}
#         hdict.update({'loglik': np.asarray(self._smoother_loglik)})
#         for i in range(self._dim_x):
#             hdict.update({f'm_{i}': np.asarray(self._smoother_mean)[:, i]})
#             for j in range(i, self._dim_x):
#                 hdict.update(
#                     {f'P_{i}{j}': np.asarray(self._filter_cov)[:, i, j]})
#             # hdict.update({f'P_{i}{i}': np.asarray(
#             #     self._smoother_cov)[:, i, i]})
#         return pd.DataFrame(hdict)

#     @property
#     def id(self) -> float:
#         """Object id"""
#         return self._id

#     @property
#     def dim_state(self) -> float:
#         """Dimension of state space """
#         return self._dim_x

#     @property
#     def dim_obs(self) -> float:
#         """Dimension of observation space """
#         return self._dim_y

#     @property
#     def K(self) -> ndarray:
#         """Kalman gain matrix at the last update """
#         return self._K

#     @property
#     def S(self) -> ndarray:
#         """Likelihood covariance matrix at the last update """
#         return self._S

#     @property
#     def time_elapsed(self) -> float:
#         """Time elapsed so far """
#         return self._time_elapsed

#     @property
#     def last_update_at(self) -> float:
#         """Time elapsed so far """
#         return self._last_update_at

#     @property
#     def time_history(self) -> float:
#         """Time history of time elapsed """
#         return np.asarray(self._time_history)

#     @property
#     def observations(self) -> float:
#         """Observations"""
#         return np.asarray(self._observations)

#     @property
#     def loglik(self) -> float:
#         """ Likelihood of the observation at the last update """
#         return self._loglik

#     @property
#     def filter_loglik(self) -> float:
#         """ time history of log-likelihood of observations """
#         return np.asarray(self._filter_loglik)

#     @property
#     def filter_mean(self) -> float:
#         """ time history of state mean """
#         return np.asarray(self._filter_mean).T

#     @property
#     def filter_cov(self) -> float:
#         """ time history of state covariance matrix """
#         return np.asarray(self._filter_cov).T

#     @property
#     def smoother_mean(self) -> float:
#         """ time history of state mean """
#         return np.asarray(self._smoother_mean).T

#     @property
#     def smoother_cov(self) -> float:
#         """ time history of state covariance matrix """
#         return np.asarray(self._smoother_cov).T

#     @property
#     def smoother_loglik(self) -> float:
#         """ time history of state covariance matrix """
#         return np.asarray(self._smoother_loglik)

#     @property
#     def m(self) -> ndarray:
#         """Get state mean"""
#         return self._m

#     @property
#     def P(self) -> ndarray:
#         """Get state covariance matrix"""
#         return self._P

#     @property
#     def Q(self) -> ndarray:
#         """Getter for process error covariance matrix"""
#         return self._Q

#     @Q.setter
#     def Q(self, val) -> None:
#         """Setter for process error covariance matrix"""
#         val = np.asarray(val) if not isinstance(val, np.ndarray) else val
#         if val.shape != self._Q.shape:
#             print('KalmanFilter: Shape mismatch while setting Q matrix!')
#             raise ValueError(f'Desired: {self._Q.shape} Input: {val.shape}')
#         self._Q = val

#     @property
#     def F(self) -> ndarray:
#         """State transition matrix"""
#         return self._F

#     @F.setter
#     def F(self, val) -> None:
#         val = np.asarray(val) if not isinstance(val, np.ndarray) else val
#         if val.shape != self._F.shape:
#             print('KalmanFilter: Shape mismatch while setting F matrix!')
#             raise ValueError(f'Desired: {self._F.shape} Input: {val.shape}')
#         self._F = val

#     @property
#     def H(self) -> ndarray:
#         """Getter for observation-state relation"""
#         return self._H

#     @H.setter
#     def H(self, val) -> None:
#         """Setter for observation-state relation"""
#         val = np.asarray(val) if not isinstance(val, np.ndarray) else val
#         if val.shape[1] != self._dim_x:
#             print('KalmanFilter: Shape mismatch while setting H matrix!')
#             raise ValueError(f'Desired: {self._H.shape} Input: {val.shape}')
#         self._H = val

#     @property
#     def R(self) -> ndarray:
#         """Getter for observation error covariance matrix"""
#         return self._R

#     @R.setter
#     def R(self, val) -> None:
#         """Setter for observation error covariance matrix"""
#         val = np.asarray(val) if not isinstance(val, np.ndarray) else val
#         # if val.shape[1] != self._R.shape:
#         #     print('KalmanFilter: Shape mismatch while setting R matrix!')
#         #     raise ValueError(f'Desired: {self._F.shape} Input: {val.shape}')
#         self._R = val
