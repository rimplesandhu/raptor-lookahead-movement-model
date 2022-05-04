""" Kalman filter base class """
from collections.abc import Sequence, Callable
from abc import ABC, abstractmethod
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
        name: str,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0
    ):
        # basic info
        self.name: str = name
        self.id: str = str(object_id)
        self.dt: float = dt

        # private variables
        self._nx: int = int(nx)  # dimension of state vector
        self._ny: int = int(ny)  # dimension of observation vector
        self._m: ndarray | None = None  # state mean vector
        self._P: ndarray | None = None  # state covariance matrix
        self._F: ndarray | None = None  # state transition matrix
        self._Q: ndarray | None = None  # process error cov mat
        self._H: ndarray | None = None  # observation function
        self._R: ndarray | None = None  # obs error covariance matrix
        self._K: ndarray | None = None  # kalman gain mat
        self._S: ndarray | None = None  # innovation matrix
        self._f: Callable | None = None  # state transition function
        self._h: Callable | None = None  # measurement-state function
        self._nees: float = np.nan  # normalized estimation error squared
        self._nis: float = np.nan  # normalized innovation squared
        self._loglik: float = np.nan  # log lik of observations
        self._time_elapsed: float = 0.  # time elapsed, default starts at 0.
        self._last_update_at: float = 0.  # last update at this time
        self._labels: Sequence[str] = [f'x_{i}' for i in range(self._nx)]

        # saving history of filter and smoother
        col_names = ['time', 'obs', 'state_mean', 'state_cov',
                     'nees', 'nis', 'loglik', 'rmat']
        self._history = {}
        self._history_smoother = {}
        for cname in col_names:
            self._history[cname] = []
            self._history_smoother[cname] = []

    def __str__(self) -> SyntaxWarning:
        """Print output"""
        out_str = f':::{self.name}-{self.id}:\n'
        out_str += f'Dimension of the state = {self._nx}\n'
        out_str += f'Dimension of the observations = {self._ny}\n'
        out_str += f'Time interval for forecasting = {self.dt}\n'
        return out_str

    def mat_setter(self, in_mat, to_shape) -> None:
        """Returns a valid numpy array while checking for valid shape"""
        if not isinstance(in_mat, Sequence | ndarray):
            self._raiseit(f'{in_mat} should be either a sequence or ndarray!')
        in_mat = np.asarray(in_mat)
        if in_mat.shape != to_shape:
            print('Shape mismatch!')
            self._raiseit(f'Required: {to_shape}, Input: {in_mat.shape}')
        return in_mat

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
        self._m = self.mat_setter(m0, (self._nx,))
        self._P = self.mat_setter(P0, (self._nx, self._nx))
        self._nees = np.nan
        self._nis = np.nan
        self._loglik = np.nan
        self._store_this_step()

    def forecast_upto(
        self,
        upto_time: float,
        dt_tol: float = 0.001
    ) -> None:
        """Kalman filter forecast step upto some time in future"""
        time_diff = upto_time - self.time_elapsed
        if abs(time_diff) > dt_tol:
            if time_diff < 0.:
                self._raiseit('Forecasting back in time!')
            nsteps = int(np.round(time_diff / self.dt))
            for _ in range(nsteps):
                self.forecast()

### abstract methods to be implemented by children of this base class ###

    @abstractmethod
    def validate(self) -> None:
        """Check if relevant matrices and/or functions have been initiated"""

    @abstractmethod
    def forecast(self) -> None:
        """Forecast step"""

    @abstractmethod
    def update(self, y_obs: ndarray) -> None:
        """Update step"""

    @abstractmethod
    def filter(
        self,
        tvec: ndarray,
        list_of_yobs: Sequence[ndarray],
        list_of_rmat: Sequence[ndarray] | None,
        dt_tol: float
    ) -> None:
        """
        Run filtering assuming model/measurement equations are time invariant
        """

    @abstractmethod
    def smoother(self) -> None:
        """
        Run smoothing assuming model/measurement equations are time invariant
        """

### access filter/smoother results in the form of pandas dataframe ###

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
            'nis': idf['nis'].dropna().sum(),
            'nees': idf['nees'].dropna().sum(),
            'loglik': idf['loglik'].dropna().sum()
        }
        return out_dict

    @property
    def metrics_smoother(self) -> dict:
        """Get the performance metrics"""
        idf = self.sdf
        out_dict = {
            'nis': idf['nis'].dropna().sum(),
            'nees': idf['nees'].dropna().sum(),
            'loglik': idf['loglik'].dropna().sum()
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
            self._raiseit(f'Choose state_index < {self._nx}')
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

    def _raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.name}: {outstr}')

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

    def _store_this_step(
        self,
        in_yobs: ndarray | None = None
    ) -> None:
        """Store this forecast/update step"""
        if in_yobs is not None:
            self._remove_last_entry()
            self._last_update_at = self._time_elapsed
        self._history['obs'].append(in_yobs)
        self._time_elapsed = np.around(self._time_elapsed, 4)
        self._history['time'].append(self._time_elapsed)
        self._history['state_mean'].append(self._m)
        self._history['state_cov'].append(self._P)
        self._history['rmat'].append(self._R)
        self._history['nees'].append(self._nees)
        self._history['nis'].append(self._nis)
        self._history['loglik'].append(self._loglik)

    def _check_valid_obs(self, y_obs: ndarray) -> ndarray:
        """Check the validity of the observation"""
        if (y_obs.size != self._H.shape[0]) or (y_obs.ndim != 1):
            self._raiseit(f'Incompatible observation shape: {y_obs.shape}!')


### Getter for private class variables at last update/forecast###


    @property
    def nx(self) -> int:
        """Dimension of state space"""
        return self._nx

    @property
    def ny(self) -> int:
        """Dimension of observation space """
        return self._ny

    @property
    def m(self) -> ndarray:
        """State mean"""
        return self._m

    @property
    def P(self) -> ndarray:
        """State covariance matrix"""
        return self._P

    @property
    def Q(self) -> ndarray:
        """Process error covariance matrix"""
        return self._Q

    @property
    def R(self) -> ndarray:
        """Measurement error covariance matrix"""
        return self._R

    @property
    def F(self) -> ndarray:
        """State transition matrix"""
        return self._F

    @property
    def H(self) -> ndarray:
        """Measurement matrix"""
        return self._H

    @property
    def f(self) -> ndarray:
        """State transition equation"""
        return self._f

    @property
    def h(self) -> ndarray:
        """Measurement equation"""
        return self._h

    @property
    def K(self) -> ndarray:
        """Kalman gain matrix at the last update"""
        return self._K

    @property
    def S(self) -> ndarray:
        """Innovationd covariance matrix at the last update"""
        return self._S

    @property
    def time_elapsed(self) -> float:
        """Time elapsed so far """
        return self._time_elapsed

    @property
    def last_update_at(self) -> float:
        """Time elapsed so far """
        return self._last_update_at

    @property
    def nis(self) -> float:
        """Normalized innovation squared at the last update"""
        return self._nis

    @property
    def nees(self) -> float:
        """Normalized estimation error squared at the last update"""
        return self._nees

    @property
    def loglik(self) -> float:
        """Normalized estimation error squared at the last update"""
        return self._loglik

    @property
    def labels(self) -> list[str]:
        """Labels for state"""
        return list(self._labels)


### Setter for private class variables###

    @F.setter
    def F(self, in_mat: ndarray) -> None:
        """Setter for state transition matrix"""
        self._F = self.mat_setter(in_mat, (self.nx, self.nx))

    @H.setter
    def H(self, in_mat: ndarray) -> None:
        """Setter for observation-state relation"""
        self._H = self.mat_setter(in_mat, (self.ny, self.nx))

    @Q.setter
    def Q(self, in_mat: ndarray) -> None:
        """Setter for process error covariance matrix"""
        self._Q = self.mat_setter(in_mat, (self.nx, self.nx))

    @R.setter
    def R(self, in_mat: ndarray) -> None:
        """Setter for observation error covariance matrix"""
        self._R = self.mat_setter(in_mat, (self.ny, self.ny))

    @f.setter
    def f(self, in_f) -> None:
        """Setter for state transition equation"""
        if not isinstance(in_f, Callable):
            self._raiseit('f needs to be a function (callable)!')
        self._f = in_f

    @h.setter
    def h(self, in_h) -> None:
        """Setter for measurement - state equation"""
        if not isinstance(in_h, Callable):
            self._raiseit('g needs to be a function (callable)!')
        self._h = in_h
