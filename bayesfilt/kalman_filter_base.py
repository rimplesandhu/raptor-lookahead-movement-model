""" Kalman filter base class """
from collections.abc import Sequence, Callable
from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


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
        object_id: str | int = 0,
        dt_tol: float = 0.001
    ):
        # basic info
        self.name: str = 'KFBase'
        self.id: str = str(object_id)
        self.dt: float = dt
        self.dt_tol: float = dt_tol
        self._nx: int = self.int_setter(nx)  # dimension of state vector
        self._ny: int = self.int_setter(ny)  # dimension of observation vector
        self._m: ndarray | None = None  # state mean vector
        self._P: ndarray | None = None  # state covariance matrix
        self._truth: ndarray | None = None
        self._obs: ndarray | None = None

        # dynamics model
        self.f: Callable | None = None  # state transition function
        self._F: ndarray | None = None  # state transition matrix
        self._G: ndarray | None = None  # state transition matrix
        self._Q: ndarray | None = None  # process error cov mat
        self._qbar: ndarray = np.zeros((self.nx,))  # process error cov mat

        # observation model
        self.h: Callable | None = None  # observation-state function
        self._H: ndarray | None = None  # observation-state matrix
        self._J: ndarray | None = None  # observation error matrix
        self._R: ndarray | None = None  # obs error covariance matrix
        self._rbar: ndarray = np.zeros((self.ny,))  # obs error mean

        # Jacobian functions
        self.compute_F: Callable | None = None
        self.compute_G: Callable | None = None
        self.compute_Q: Callable | None = None
        self.compute_H: Callable | None = None
        self.compute_J: Callable | None = None
        self.compute_R: Callable | None = None

        # Kalman filtering parameters
        self._K: ndarray | None = None  # kalman gain mat
        self._S: ndarray | None = None  # innovation matrix
        self._nees: float = np.nan  # normalized estimation error squared
        self._nis: float = np.nan  # normalized innovation squared
        self._loglik: float = np.nan  # log lik of observations
        self._time_elapsed: float = 0.  # time elapsed, default starts at 0.
        self._last_update_at: float = 0.  # last update at this time
        self.labels: Sequence[str] = [f'x_{i}' for i in range(self._nx)]

        # saving history of filter and smoother
        col_names = ['time', 'obs', 'state_mean', 'state_cov',
                     'nees', 'nis', 'loglik', 'rmat', 'truth']
        self._history = {}
        self._history_smoother = {}
        for cname in col_names:
            self._history[cname] = []
            self._history_smoother[cname] = []

    def initiate_state(
        self,
        t0: float,
        m0: ndarray,
        P0: ndarray
    ) -> None:
        """Initiate the time, state mean and covariance matrix"""
        self._erase_history_filter()
        self._erase_history_smoother()
        self._time_elapsed = t0
        self._m = self.vec_setter(m0, self._nx)
        self._P = self.mat_setter(P0, (self._nx, self._nx))
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
                self.raiseit('Forecasting back in time!')
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
        nsteps = len(list_of_time)
        if len(list_of_obs) != nsteps:
            self.raiseit('Length mismatch: time vs list of observations')
        if list_of_R is not None:
            if len(list_of_R) != nsteps:
                self.raiseit('Length mismatch: time vs list of R matrices')
        if list_of_truth is not None:
            if len(list_of_truth) != nsteps:
                self.raiseit('Size mismatch: time vs list of state truths')
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

    def smoother(self):
        # pylint: disable=too-many-locals
        """Run smoothing assuming model/measurement eq are time invariant"""
        nsteps = len(self.history['time'])
        if nsteps < 1:
            self.raiseit('No filter history found, run filter first!')
        self._erase_history_smoother()
        for i in reversed(range(len(self.history['time']))):
            self._load_this_step(i)
            if i != len(self.history['time']) - 1:
                self.backward_update()
            self._store_this_step_smoother()
        for v in self._history_smoother.values():
            v.reverse()

    # def is_pos_def(inA: ndarray):
    #     """Returns true when input is positive-definite, via Cholesky"""
    #     try:
    #         _ = np.linalg.cholesky(B)
    #         return True
    #     except np.linalg.LinAlgError:
    #         return False

### abstract methods to be implemented by children of this base class ###

    @abstractmethod
    def validate(self) -> None:
        """Check if relevant matrices and/or functions have been initiated"""
        if (self.P is None) or (self.m is None):
            self.raiseit('Need to initiate state, use initiate_state()')
        if self.R is None:
            self.raiseit('Need to initiate R matrix!')

    @abstractmethod
    def forecast(self) -> None:
        """Forecast step"""
        self._time_elapsed += self.dt
        self._truth = None
        self._obs = None
        self._nis = np.nan
        self._nees = np.nan
        self._loglik = np.nan

    @abstractmethod
    def update(self) -> None:
        """Update step"""

    @abstractmethod
    def backward_update(self) -> None:
        """Backward Update step"""


### accessing filter/smoother results###

    @property
    def df(self) -> pd.DataFrame:
        """Pandas Dataframe containing entire history of the filter"""
        nsteps = len(self._history['time'])
        idf = pd.DataFrame(self._history)
        idf['object_id'] = np.asarray([self.id] * nsteps)
        return idf

    @property
    def sdf(self) -> pd.DataFrame:
        """Pandas Dataframe containing entire history of the smoother"""
        nsteps = len(self._history_smoother['time'])
        idf = pd.DataFrame(self._history_smoother)
        idf['object_id'] = np.asarray([self.id] * nsteps)
        return idf

    def get_state_mean(self, i: int) -> ndarray:
        """Get state mean time series of ith state"""
        return np.stack(self.df.state_mean.values)[:, i]

    def get_state_var(self, i: int, j: int | None = None) -> ndarray:
        """Get state var of ith state"""
        j = j if j is not None else i
        return np.stack(self.df.state_cov.values)[:, i, j]

    @property
    def history(self) -> dict:
        """Time history of time elapsed """
        return self._history

    @property
    def history_smoother(self) -> dict:
        """Observations"""
        return self._history_smoother

    @property
    def metrics(self) -> dict:
        """Get the performance metrics"""
        idf = self.df
        out_dict = {
            'nis': np.around(idf['nis'].dropna().sum(), 3),
            'nees': np.around(idf['nees'].dropna().sum(), 3),
            'loglik': np.around(idf['loglik'].dropna().sum(), 3)
        }
        return out_dict

    @property
    def metrics_smoother(self) -> dict:
        """Get the performance metrics"""
        idf = self.sdf
        out_dict = {
            'nis': np.around(idf['nis'].dropna().sum(), 3),
            'nees': np.around(idf['nees'].dropna().sum(), 3),
            'loglik': np.around(idf['loglik'].dropna().sum(), 3)
        }
        return out_dict

### Plotting related ###

    def plot_metric(
        self,
        in_ax: plt.Axes,
        mname: str = 'nis',
        ftype: str = 'filter',
        **kwargs
    ) -> None:
        """Plot the time history of a performance metric"""
        idf = self.sdf if ftype == 'smoother' else self.df
        idfshort = idf[~idf[mname].isna()]
        tvec = idfshort['time'].values
        zvec = idfshort[mname].values
        in_ax.plot(tvec, zvec, **kwargs)
        in_ax.set_ylabel(mname)
        in_ax.set_xlim([tvec[0], tvec[-1]])
        in_ax.set_xlabel('Time elapsed (Seconds)')

    def plot_state(
        self,
        in_ax: plt.Axes,
        indx: int,
        lcolor: str = 'r',
        ftype: str = 'filter',
        cb_fac: float = 3.,
        **kwargs
    ) -> None:
        """Plot the time history of state estimates"""
        if indx > self._nx:
            self.raiseit(f'Choose state_index < {self._nx}')
        idf = self.sdf if ftype == 'smoother' else self.df
        tvec = idf['time'].values
        zvec = np.stack(idf['state_mean'].values)[:, indx]
        zvec_cb = cb_fac * \
            np.stack(idf['state_cov'].values)[:, indx, indx]**0.5
        in_ax.plot(tvec, zvec, color=lcolor, **kwargs)
        in_ax.fill_between(tvec, zvec - zvec_cb, zvec + zvec_cb,
                           fc=lcolor, ec='none', alpha=0.25)
        in_ax.set_ylabel(f'{self.labels[indx]}')
        in_ax.set_xlim([tvec[0], tvec[-1]])
        in_ax.set_xlabel('Time elapsed (Seconds)')


### Private class functions ###


    def _get_label_for_this(self, idx: int) -> tuple[str, str]:
        """Get labels for mean and variance for this state index"""
        return (self.labels[idx], f'var_{self.labels[idx]}')

    def _erase_history_filter(self) -> None:
        """Erase history of filter"""
        for v in self._history.values():
            v.clear()

    def _erase_history_smoother(self) -> None:
        """Erase history of smoother"""
        for v in self._history_smoother.values():
            v.clear()

    def _remove_last_entry(self) -> None:
        """Remove last entry in the record keeping"""
        for v in self._history.values():
            del v[-1]

    def _store_this_step(self, update: bool = False) -> None:
        """Store this forecast/update step"""
        if update:
            self._remove_last_entry()
            self._last_update_at = self.time_elapsed
        self._time_elapsed = np.around(self.time_elapsed, 4)
        self._history['obs'].append(self.obs)
        self._history['truth'].append(self.truth)
        self._history['time'].append(self.time_elapsed)
        self._history['state_mean'].append(deepcopy(self.m))
        self._history['state_cov'].append(deepcopy(self.P))
        self._history['rmat'].append(self.R)
        self._history['nees'].append(self.nees)
        self._history['nis'].append(self.nis)
        self._history['loglik'].append(self.loglik)

    def _store_this_step_smoother(self) -> None:
        """Store this smoother step"""
        self._time_elapsed = np.around(self.time_elapsed, 4)
        self._history_smoother['time'].append(self.time_elapsed)
        self._history_smoother['obs'].append(self.obs)
        self._history_smoother['truth'].append(self.truth)
        self._history_smoother['state_mean'].append(self.m)
        self._history_smoother['state_cov'].append(self.P)
        self._history_smoother['rmat'].append(self.R)
        self._history_smoother['nees'].append(self.nees)
        self._history_smoother['nis'].append(self.nis)
        self._history_smoother['loglik'].append(self.loglik)

    def _load_this_step(self, i: int) -> None:
        """Load this forecast/update step"""
        self._time_elapsed = self.history['time'][i]
        self.obs = self.history['obs'][i]
        self.truth = self.history['truth'][i]
        self._m = deepcopy(self.history['state_mean'][i])
        self._P = deepcopy(self.history['state_cov'][i])
        self.R = self.history['rmat'][i].copy()
        self._nis = self.history['nis'][i]
        self._nees = self.history['nees'][i]
        self._loglik = self.history['loglik'][i]

### Matrix/vector handling###

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f':::{self.name}-{self.id}:\n'
        out_str += f'Dimension of the state = {self._nx}\n'
        out_str += f'Dimension of the observations = {self._ny}\n'
        out_str += f'Time interval for forecasting = {self.dt}\n'
        return out_str

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

    @ property
    def K(self) -> ndarray:
        """Kalman gain matrix at the last update"""
        return self._K

    @ property
    def S(self) -> ndarray:
        """Innovationd covariance matrix at the last update"""
        return self._S

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
        """Setter for observations"""
        self._obs = self.vec_setter(
            in_mat, self.ny) if in_mat is not None else None

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
    def symmetrize(a_mat: ndarray) -> ndarray:
        """Return a symmetrized version of NumPy array"""
        return (a_mat + a_mat.T) / 2.
