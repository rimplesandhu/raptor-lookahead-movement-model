""" Kalman filter class """
from typing import Tuple, Optional, Union
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


class KalmanFilter:
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods
    # pylint: disable=invalid-name
    """ Class for implementing Kalman Filter """

    def __init__(self,
                 dim_x: int,
                 dim_y: int = 1,
                 object_id: Union[str, int] = 0,
                 dt_tol: float = 0.001
                 ):
        self._id = str(object_id)
        self._dim_x: int = int(dim_x) if isinstance(dim_x, float) else dim_x
        self._dim_y: int = int(dim_y) if isinstance(dim_x, float) else dim_y
        self._m: ndarray = np.zeros(dim_x)  # state mean vector
        self._P: ndarray = np.eye(dim_x)  # state covariance matrix
        self._F: ndarray = np.eye(dim_x)  # state transition matrix
        self._Q: ndarray = np.eye(dim_x)  # model error covariance matrix
        self._H: ndarray = np.zeros((dim_y, dim_x))  # observation function
        self._R: ndarray = np.eye(dim_y)  # obs error covariance matrix
        self._K: ndarray = np.zeros((dim_y, dim_x))  # kalman gain matrix
        self._S: ndarray = np.zeros((dim_y, dim_y))  # innovation matrix
        self._loglik: float = 0.  # likelihood of data given predicted state
        self._time_elapsed: float = 0.  # time elapsed, default starts at 0.
        self._last_update_at: float = 0.  # last update at this time
        self._dt_tolerance: float = dt_tol
        self._time_history = []
        self._filter_mean = []
        self._filter_cov = []
        self._filter_loglik = []
        self._smoother_mean = []
        self._smoother_cov = []
        self._smoother_loglik = []
        self._observations = []
        self.labels = [f'x_{i}' for i in range(self._dim_x)]

    def filter(
        self,
        dt_pred: float,
        tobs: ndarray,
        yobs: ndarray,
        yobs_var: ndarray
    ) -> None:
        """
        Run filtering assuming F, H, Q matrices are time invariant
        """
        k = 1
        self._time_elapsed = tobs[0]
        while k < tobs.size:
            if self._time_elapsed - tobs[k] > self._dt_tolerance:
                raise Exception('Skipping an observation, lower the dt!')
            if abs(self._time_elapsed - tobs[k]) < self._dt_tolerance:
                np.fill_diagonal(self._R, yobs_var[k, :])
                self.update(yobs[k, :])
                k += 1
            else:
                self.forecast(dt_pred)

    def forecast_upto(self, upto_time: float, time_dt: float) -> None:
        """
        Kalman filter forecast step upto some time in future
        """
        time_diff = upto_time - self.time_elapsed
        if abs(time_diff) > self._dt_tolerance:
            if time_diff < 0.:
                raise Exception('KF: Forecasting back in time!')
            nsteps = int(np.round(time_diff / time_dt))
            # print(f'going from {self.time_elapsed} to {upto_time} in {nsteps}')
            for _ in range(nsteps):
                self.forecast(time_dt)

    def forecast(self, time_dt: Optional[float] = None) -> None:
        """
        Kalman filter forecast step
        """
        self._m = np.dot(self._F, self._m)
        self._P = np.matmul(self._F, np.matmul(self._P, self._F.T)) + self._Q
        self._time_elapsed += time_dt if time_dt is not None else 1.
        self._loglik = np.nan
        self._store_this_step()

    def update(self, y_obs: ndarray) -> None:
        """
        Kalman filter update step
        """
        y_obs = self.get_valid_obs(y_obs)
        y_pred = self._H @ self._m
        _residual = y_obs - y_pred
        self._S = self._H @ self._P @ self._H.T + self._R
        _s_inv = np.linalg.pinv(self._S, hermitian=True)
        self._loglik = self.get_loglik_of_obs(y_obs)
        self._K = self._P @ self._H.T @ _s_inv
        self._m = self._m + np.dot(self._K, _residual)
        self._P = (np.eye(self._dim_x) - self._K @ self._H) @ self._P
        self._store_this_step(obs=y_obs)

    def initiate(self, in_time: float, in_m: ndarray, in_pmat: ndarray):
        """
        Initiate the state mean and covariance matrix
        """
        in_m = np.asarray(in_m) if not isinstance(in_m, ndarray) else in_m
        in_pmat = np.asarray(in_pmat) if not isinstance(
            in_pmat, ndarray) else in_pmat
        assert in_m.ndim == 1, 'KF: Need 1D mean vector'
        assert in_pmat.ndim == 2, 'KF: Need 2D covaince matrix'
        assert in_m.size == self._dim_x, 'KF: Wrong size for mean vector!'
        assert in_pmat.shape == (
            self._dim_x, self._dim_x), 'KF: Wrong size for mean vector!'
        self._m = in_m
        self._P = in_pmat
        self._time_elapsed = in_time
        self._loglik = np.nan
        self._store_this_step(first_update=True)

    def get_loglik_of_obs(self, y_obs: ndarray) -> None:
        """
        Compute log-likelihood of this observation
        """
        y_obs = self.get_valid_obs(y_obs)
        y_pred = self._H @ self._m
        # print(y_pred)
        _residual = y_obs - y_pred
        _this_smat = self._H @ self._P @ self._H.T + self._R
        _s_inv = np.linalg.pinv(_this_smat, hermitian=True)
        # _s_inv = np.linalg.inv(_this_smat)
        # print(_s_inv.diagonal())
        this_loglik = self._dim_y * np.log(2. * np.pi)
        this_loglik += np.log(np.linalg.det(_this_smat))
        this_loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
        this_loglik = np.linalg.multi_dot([_residual.T, _s_inv, _residual])
        # this_loglik = np.dot(_residual, np.dot(_s_inv, _residual))
        # this_loglik = np.dot(_residual, _residual)
        this_loglik *= -0.5
        return this_loglik

    def get_valid_obs(self, y_obs: ndarray) -> ndarray:
        """Get observations in a compatible form and also check validity"""
        y_obs = np.asarray(y_obs)
        y_obs = y_obs.flatten()
        assert y_obs.size == self._H.shape[0], 'Incompatible obs with H matrix'
        assert y_obs.size == self._R.shape[0], 'Incompatible obs with R matrix'
        assert y_obs.size == self._R.shape[1], 'Incompatible obs with R matrix'
        return y_obs

    def smoother(self):
        """ Run smoothing assuming F, H, Q matrices are time invariant """
        num_steps = np.asarray(self._filter_mean).shape[0]
        self._smoother_mean = []
        self._smoother_cov = []
        self._smoother_loglik = []
        for i in reversed(range(num_steps)):
            if i == num_steps - 1:
                _smean = self._filter_mean[i]
                _scov = self._filter_cov[i]
            else:
                _umean = self._filter_mean[i]
                _ucov = self._filter_cov[i]
                _fcov = self._F @ _ucov @ self._F.T + self._Q
                _fcov_inv = np.linalg.pinv(_fcov)
                _gmat = _ucov @ self._F.T @ _fcov_inv
                _smean = _umean + _gmat @ (_smean - self._F @ _umean)
                _scov = _ucov + _gmat @ (_scov - _fcov) @ _gmat.T
            self._smoother_mean.append(_smean)
            self._smoother_cov.append(_scov)
        self._smoother_mean.reverse()
        self._smoother_cov.reverse()
        self._compute_loglik_smoother()

    def _compute_loglik_smoother(self):
        """ Computes loglik of observations given smooothened estimates """
        k = 1
        for i, _floglik in enumerate(self._filter_loglik):
            if not np.isnan(_floglik):
                _scov_prev = self._smoother_cov[i - 1]
                _ypred = self._H @ self._F @ self._smoother_mean[i - 1]
                _residual = self._observations[k, :] - _ypred
                _smat = self._H @ _scov_prev @ self._H.T + self._R
                _s_inv = np.linalg.pinv(_smat, hermitian=True)
                _loglik = self._dim_y * np.log(2. * np.pi)
                _loglik += np.log(np.linalg.det(_smat))
                _loglik += np.linalg.multi_dot([_residual.T,
                                                _s_inv, _residual])
                _loglik *= -0.5
                k += 1
            else:
                _loglik = np.nan
            self._smoother_loglik.append(_loglik)

    def _store_this_step(self, obs: Optional[ndarray] = None,
                         first_update=False):
        if not first_update:
            time_diff = self._time_history[-1] - self.time_elapsed
            if abs(time_diff) < self._dt_tolerance:
                self._remove_last_entry()
        else:
            self._last_update_at = self._time_elapsed
        self._filter_loglik.append(self._loglik)
        self._time_history.append(self._time_elapsed)
        self._filter_mean.append(self._m)
        self._filter_cov.append(self._P)
        if obs is not None:
            self._observations.append(obs)
            self._last_update_at = self._time_elapsed

    def _remove_last_entry(self) -> None:
        """ Remove last entry in the record keeping """
        del self._time_history[-1]
        del self._filter_mean[-1]
        del self._filter_cov[-1]
        del self._filter_loglik[-1]

    def plot_loglik(
        self,
        time_window: Optional[Tuple[float, float]] = None
    ):
        """ plots the time history of log-likelihood of observations """
        if len(self._time_history) < 1:
            raise ValueError('KF: No state history found, run filter first!')
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fdf = self.df
        fdfshort = fdf[~fdf['loglik'].isna()]
        ax.plot(fdfshort['time_elapsed'], fdfshort['loglik'], '.b')
        ax.set_ylabel('Log-likelihood of obs')
        ax.grid(True)
        if time_window is None:
            ax.set_xlim([fdfshort['time_elapsed'].iloc[0],
                         fdfshort['time_elapsed'].iloc[-1]])
        else:
            ax.set_xlim(time_window)
        ax.set_xlabel('Time elapsed (Seconds)')
        fig.tight_layout()
        return fig, ax

    def plot_filtered_state(
        self,
        ax,
        state_index: int,
        lcolor='b',
        time_window: Optional[Tuple[float, float]] = None,
        cb_fac=3.
    ):
        """ plots the time history of state estimates """
        assert state_index < self._dim_x, f'Choose state_index < {self._dim_x}'
        m_name, p_name = self.get_label_for_this(state_index)
        idf = self.df.copy()
        t_history = idf['time_elapsed'].values / 60.
        m_history = idf[m_name].values
        cb_history = cb_fac * idf.loc[:, p_name].values**0.5
        ax.plot(t_history, m_history, linestyle='-', color=lcolor,
                label='Kalman Filter')
        ax.fill_between(t_history, m_history - cb_history,
                        m_history + cb_history,
                        fc=lcolor, ec='none', alpha=0.25)
        ax.set_ylabel(f'{self.labels[state_index]}')
        if time_window is None:
            ax.set_xlim([t_history[0], t_history[-1]])
        else:
            ax.set_xlim(time_window)
        ax.set_xlabel('Time elapsed (minutes)')

    def plot_smoother_state(
        self,
        ax,
        state_index: int,
        lcolor='b',
        time_window: Optional[Tuple[float, float]] = None,
        cb_fac=3.
    ):
        """ plots the time history of state estimates """
        assert state_index < self._dim_x, f'Choose state_index < {self._dim_x}'
        m_name, p_name = self.get_label_for_this(state_index)
        idf = self.dfs.copy()
        if idf.shape != self.df.shape:
            self.smoother()
            idf = self.dfs.copy()
        t_history = idf['time_elapsed'].values / 60.
        m_history = idf[m_name].values
        cb_history = cb_fac * idf.loc[:, p_name].values**0.5
        ax.plot(t_history, m_history, linestyle='-', color=lcolor,
                label='Kalman Smoother')
        ax.fill_between(t_history, m_history - cb_history,
                        m_history + cb_history,
                        fc=lcolor, ec='none', alpha=0.25)
        ax.set_ylabel(f'{self.labels[state_index]}')
        if time_window is None:
            ax.set_xlim([t_history[0], t_history[-1]])
        else:
            ax.set_xlim(time_window)
        ax.set_xlabel('Time elapsed (minutes)')

    def plot_observations(
        self,
        ax,
        obs_index: int,
        lcolor='r',
    ):
        """ plots the time history of state estimates """
        idf = self.df.copy()
        iudf = idf[~idf['loglik'].isna()]
        ax.plot(iudf['time_elapsed'] / 60., self.observations[:, obs_index],
                '*', color=lcolor, mec=None, alpha=0.5,
                markersize=5., label='Observations')

    def get_label_for_this(self, idx: int):
        """Get labels for this state index"""
        return (self.labels[idx], f'var_{self.labels[idx]}')

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe containing entire history of the filtering"""
        hdict = {'object_id': np.asarray([self._id] * len(self._time_history))}
        hdict.update({'time_elapsed': np.asarray(self._time_history)})
        hdict.update({'loglik': np.asarray(self._filter_loglik)})
        for i in range(self._dim_x):
            m_name, p_name = self.get_label_for_this(i)
            hdict.update({m_name: np.asarray(self._filter_mean)[:, i]})
            hdict.update({p_name: np.asarray(self._filter_cov)[:, i, i]})
        return pd.DataFrame(hdict)

    @property
    def dfs(self) -> pd.DataFrame:
        """Dataframe containing entire history of the smoothing"""
        hdict = {'time_elapsed': np.asarray(self._time_history)}
        hdict.update({'loglik': np.asarray(self._smoother_loglik)})
        for i in range(self._dim_x):
            hdict.update({f'm_{i}': np.asarray(self._smoother_mean)[:, i]})
            for j in range(i, self._dim_x):
                hdict.update(
                    {f'P_{i}{j}': np.asarray(self._filter_cov)[:, i, j]})
            # hdict.update({f'P_{i}{i}': np.asarray(
            #     self._smoother_cov)[:, i, i]})
        return pd.DataFrame(hdict)

    @property
    def id(self) -> float:
        """Object id"""
        return self._id

    @property
    def dim_state(self) -> float:
        """Dimension of state space """
        return self._dim_x

    @property
    def dim_obs(self) -> float:
        """Dimension of observation space """
        return self._dim_y

    @property
    def K(self) -> ndarray:
        """Kalman gain matrix at the last update """
        return self._K

    @property
    def S(self) -> ndarray:
        """Likelihood covariance matrix at the last update """
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
    def time_history(self) -> float:
        """Time history of time elapsed """
        return np.asarray(self._time_history)

    @property
    def observations(self) -> float:
        """Observations"""
        return np.asarray(self._observations)

    @property
    def loglik(self) -> float:
        """ Likelihood of the observation at the last update """
        return self._loglik

    @property
    def filter_loglik(self) -> float:
        """ time history of log-likelihood of observations """
        return np.asarray(self._filter_loglik)

    @property
    def filter_mean(self) -> float:
        """ time history of state mean """
        return np.asarray(self._filter_mean).T

    @property
    def filter_cov(self) -> float:
        """ time history of state covariance matrix """
        return np.asarray(self._filter_cov).T

    @property
    def smoother_mean(self) -> float:
        """ time history of state mean """
        return np.asarray(self._smoother_mean).T

    @property
    def smoother_cov(self) -> float:
        """ time history of state covariance matrix """
        return np.asarray(self._smoother_cov).T

    @property
    def smoother_loglik(self) -> float:
        """ time history of state covariance matrix """
        return np.asarray(self._smoother_loglik)

    @property
    def m(self) -> ndarray:
        """Get state mean"""
        return self._m

    @property
    def P(self) -> ndarray:
        """Get state covariance matrix"""
        return self._P

    @property
    def Q(self) -> ndarray:
        """Getter for process error covariance matrix"""
        return self._Q

    @Q.setter
    def Q(self, val) -> None:
        """Setter for process error covariance matrix"""
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._Q.shape:
            print('KalmanFilter: Shape mismatch while setting Q matrix!')
            raise ValueError(f'Desired: {self._Q.shape} Input: {val.shape}')
        self._Q = val

    @property
    def F(self) -> ndarray:
        """State transition matrix"""
        return self._F

    @F.setter
    def F(self, val) -> None:
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._F.shape:
            print('KalmanFilter: Shape mismatch while setting F matrix!')
            raise ValueError(f'Desired: {self._F.shape} Input: {val.shape}')
        self._F = val

    @property
    def H(self) -> ndarray:
        """Getter for observation-state relation"""
        return self._H

    @H.setter
    def H(self, val) -> None:
        """Setter for observation-state relation"""
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape[1] != self._dim_x:
            print('KalmanFilter: Shape mismatch while setting H matrix!')
            raise ValueError(f'Desired: {self._H.shape} Input: {val.shape}')
        self._H = val

    @property
    def R(self) -> ndarray:
        """Getter for observation error covariance matrix"""
        return self._R

    @R.setter
    def R(self, val) -> None:
        """Setter for observation error covariance matrix"""
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        # if val.shape[1] != self._R.shape:
        #     print('KalmanFilter: Shape mismatch while setting R matrix!')
        #     raise ValueError(f'Desired: {self._F.shape} Input: {val.shape}')
        self._R = val
