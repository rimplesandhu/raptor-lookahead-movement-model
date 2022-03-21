""" Kalman filter class """
from typing import Tuple, Optional
import numpy as np
from numpy import ndarray
import pandas as pd
import matplotlib.pyplot as plt


class KalmanFilter:
    """ Class for implementing Kalman Filter """

    def __init__(self, dim_x: int, dim_y: int):
        self._dim_x: int = int(dim_x) if isinstance(dim_x, float) else dim_x
        self._dim_y: int = int(dim_y) if isinstance(dim_x, float) else dim_y
        self._m: ndarray = np.zeros(dim_x)  # state mean vector
        self._P: ndarray = np.eye(dim_x)  # state covariance matrix
        self._F: ndarray = np.eye(dim_x)  # state transition matrix
        self._Q: ndarray = np.eye(dim_x)  # model error covariance matrix
        self._H: ndarray = np.zeros((dim_y, dim_x))  # observation function
        self._R: ndarray = np.eye(dim_y)  # observation error covariance matrix
        self._K: ndarray = np.zeros((dim_y, dim_x))  # kalman gain matrix
        self._S: ndarray = np.zeros((dim_y, dim_y))  # kalman gain matrix
        self._loglik: float = 0.  # likelihood of data given predicted state
        self._time_elapsed = 0.  # time elapsed, default starts at 0.
        self._time_history = []
        self._filter_mean = []
        self._filter_cov = []
        self._filter_loglik = []
        self._smoother_mean = []
        self._smoother_cov = []
        self._smoother_loglik = []
        self._observations = []
        self.labels = [f'x_{i}' for i in range(self._dim_x)]
        self._is_filtering_done = False
        self._is_smoothing_done = False
        self._is_first_update_done = False

    def filter(
        self,
        dt_pred: float,
        tobs: ndarray,
        yobs: ndarray,
        yobs_var: ndarray
    ) -> None:
        """ Run filtering assuming F, H, Q matrices are time invariant """
        k = 1
        self._time_elapsed = tobs[0]
        while k < tobs.size:
            if abs(self._time_elapsed - tobs[k]) < 0.01:
                np.fill_diagonal(self._R, yobs_var[k, :])
                self.update(yobs[k, :])
                k += 1
                self._is_first_update_done = True
            else:
                self.forecast(dt_pred)
        self._observations = yobs
        self._is_filtering_done = True

    def forecast(self, time_dt: Optional[float] = None) -> None:
        """
        Kalman filter forecast step
        """
        self._m = np.dot(self._F, self._m)
        self._P = np.matmul(self._F, np.matmul(self._P, self._F.T)) + self._Q
        self._time_elapsed += time_dt if time_dt is not None else 1
        if self._is_first_update_done:
            self._store_this_step()

    def update(self, y_obs: ndarray) -> None:
        """
        Kalman filter update step
        """
        y_obs = np.asarray(y_obs)
        y_obs = y_obs.flatten()
        if y_obs.size != self._dim_y:
            raise ValueError(f'KF: # of observations should be {self._dim_y}!')
        y_pred = self._H @ self._m
        _residual = y_obs - y_pred
        self._S = self._H @ self._P @ self._H.T + self._R
        _s_inv = np.linalg.pinv(self._S, hermitian=True)
        self._loglik = self._dim_y * np.log(2. * np.pi)
        self._loglik += np.log(np.linalg.det(self._S))
        self._loglik += np.linalg.multi_dot([_residual.T, _s_inv, _residual])
        self._loglik *= -0.5
        self._K = self._P @ self._H.T @ _s_inv
        self._m = self._m + np.dot(self._K, _residual)
        self._P = (np.eye(self._dim_x) - self._K @ self._H) @ self._P
        self._store_this_step(update_step=True)

    def smoother(self):
        """ Run smoothing assuming F, H, Q matrices are time invariant """
        if not self._is_filtering_done:
            raise ValueError('KF: Run filter before running smoother!')
        num_steps = np.asarray(self._filter_mean).shape[0]
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
        self._is_smoothing_done = True

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

    def _store_this_step(self, update_step=False):
        if update_step:
            if self._is_first_update_done:
                self._remove_last_entry()
            self._filter_loglik.append(self._loglik)
        else:
            self._filter_loglik.append(np.nan)
        self._time_history.append(self._time_elapsed)
        self._filter_mean.append(self._m)
        self._filter_cov.append(self._P)

    def _remove_last_entry(self) -> None:
        """ Remove last entry in the record keeping """
        del self._time_history[-1]
        del self._filter_mean[-1]
        del self._filter_cov[-1]
        del self._filter_loglik[-1]

    def plot_loglik(
        self,
        show_legend: bool = True,
        time_window: Optional[Tuple[float, float]] = None
    ):
        """ plots the time history of log-likelihood of observations """
        if not self._is_filtering_done:
            raise ValueError('KF: No state history found, run filter first!')
        fig, ax = plt.subplots(figsize=(12, 3.5))
        fdf = self.df_filter
        fdfshort = fdf[~fdf['loglik'].isna()]
        ax.plot(fdfshort['time_elapsed'], fdfshort['loglik'], '.b',
                label='Kalman filter')
        if self._is_smoothing_done:
            sdf = self.df_smoother
            sdfshort = sdf[~sdf['loglik'].isna()]
            ax.plot(sdfshort['time_elapsed'], sdfshort['loglik'], '.r',
                    label='Kalman Smoother')
        ax.set_ylabel('Log-likelihood of obs')
        ax.grid(True)
        if show_legend:
            ax.legend()
        if time_window is None:
            ax.set_xlim([0., fdf['time_elapsed'].values[-1]])
        else:
            ax.set_xlim(time_window)
        ax.set_xlabel('Time elapsed (Seconds)')
        fig.tight_layout()
        return fig, ax

    def plot_state_estimates(
        self,
        state_index: int,
        obs_index: Optional[int] = None,
        time_window: Optional[Tuple[float, float]] = None,
        plot_filter: bool = True,
        plot_smoother: bool = True,
        show_legend: bool = True
    ):
        """ plots the time history of state estimates """
        fig, ax = plt.subplots(figsize=(12, 3.5))
        cb_fac = 1.
        col_name_m = f'm_{state_index}'
        col_name_p = f'P_{state_index}{state_index}'
        fdf = self.df_filter
        if plot_filter:
            t_history = fdf['time_elapsed'].values
            m_history = fdf[col_name_m].values
            cb_history = cb_fac * fdf.loc[:, col_name_p].values**0.5
            ax.plot(t_history, m_history, '-b', label='Kalman Filter')
            ax.fill_between(t_history, m_history - cb_history,
                            m_history + cb_history,
                            fc='b', ec='none', alpha=.3)
        if plot_smoother:
            sdf = self.df_smoother
            assert fdf.shape == sdf.shape
            t_history = sdf['time_elapsed'].values
            m_history = sdf[col_name_m].values
            cb_history = cb_fac * sdf[col_name_p].values**0.5
            ax.plot(t_history, m_history, '-g', label='Kalman Smoother')
            ax.fill_between(t_history, m_history - cb_history,
                            m_history + cb_history,
                            fc='g', ec='none', alpha=.3)
        if obs_index is not None:
            udf = fdf[~fdf['loglik'].isna()]
            ax.plot(udf['time_elapsed'], self._observations[1:, obs_index],
                    '*r', alpha=0.5,
                    markersize=5., label='Observations')
        ax.set_ylabel(f'State: {self.labels[state_index]}')
        if show_legend:
            ax.legend()
        ax.grid(True)
        if time_window is None:
            ax.set_xlim([0., t_history[-1]])
        else:
            ax.set_xlim(time_window)
        ymin = np.amin(m_history)
        ymax = np.amax(m_history)
        ymargin = 0.4 * (ymax - ymin)
        #ax.set_ylim([ymin - ymargin, ymax + ymargin])
        ax.set_xlabel('Time elapsed (Seconds)')
        fig.tight_layout()
        return fig, ax

    @property
    def df_filter(self) -> pd.DataFrame:
        """ Dataframe containing entire history of the filtering"""
        if not self._is_filtering_done:
            raise ValueError('KF: No state history found, run filter first!')
        hdict = {'time_elapsed': np.asarray(self._time_history)}
        hdict.update({'loglik': np.asarray(self._filter_loglik)})
        for i in range(self._dim_x):
            hdict.update({f'm_{i}': np.asarray(self._filter_mean)[:, i]})
            for j in range(i, self._dim_x):
                hdict.update(
                    {f'P_{i}{j}': np.asarray(self._filter_cov)[:, i, j]})
        return pd.DataFrame(hdict)

    @property
    def df_smoother(self) -> pd.DataFrame:
        """ Dataframe containing entire history of the filtering"""
        if not self._is_smoothing_done:
            raise ValueError('KF: Need to run smoother first!')
        hdict = {'time_elapsed': np.asarray(self._time_history)}
        hdict.update({'loglik': np.asarray(self._smoother_loglik)})
        for i in range(self._dim_x):
            hdict.update({f'm_{i}': np.asarray(self._smoother_mean)[:, i]})
            for j in range(i, self._dim_x):
                hdict.update(
                    {f'P_{i}{j}': np.asarray(self._filter_cov)[:, i, j]})
        return pd.DataFrame(hdict)

    @property
    def dim_x(self) -> float:
        """ Dimension of state space """
        return self._dim_x

    @property
    def dim_y(self) -> float:
        """ Dimension of observation space """
        return self._dim_y

    @property
    def K(self) -> ndarray:
        """ Kalman gain matrix at the last update """
        return self._K

    @property
    def S(self) -> ndarray:
        """ Likelihood covariance matrix at the last update """
        return self._S

    @property
    def time_elapsed(self) -> float:
        """ Time elapsed so far """
        return self._time_elapsed

    @property
    def time_history(self) -> float:
        """ time history of time elapsed """
        return np.asarray(self._time_history)

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
        return self._m

    @m.setter
    def m(self, val) -> None:
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._m.shape:
            print('KalmanFilter: Shape mismatch while setting m vector!')
            raise ValueError(f'Desired: {self._m.shape} Input: {val.shape}')
        self._m = val

    @property
    def P(self) -> ndarray:
        return self._P

    @P.setter
    def P(self, val) -> None:
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._P.shape:
            print('KalmanFilter: Shape mismatch while setting P matrix!')
            raise ValueError(f'Desired: {self._P.shape} Input: {val.shape}')
        self._P = val

    @property
    def Q(self) -> ndarray:
        return self._Q

    @Q.setter
    def Q(self, val) -> None:
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._Q.shape:
            print('KalmanFilter: Shape mismatch while setting Q matrix!')
            raise ValueError(f'Desired: {self._Q.shape} Input: {val.shape}')
        self._Q = val

    @property
    def F(self) -> ndarray:
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
        return self._H

    @H.setter
    def H(self, val) -> None:
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._H.shape:
            print('KalmanFilter: Shape mismatch while setting H matrix!')
            raise ValueError(f'Desired: {self._H.shape} Input: {val.shape}')
        self._H = val

    @property
    def R(self) -> ndarray:
        return self._R

    @R.setter
    def R(self, val) -> None:
        val = np.asarray(val) if not isinstance(val, np.ndarray) else val
        if val.shape != self._R.shape:
            print('KalmanFilter: Shape mismatch while setting R matrix!')
            raise ValueError(f'Desired: {self._F.shape} Input: {val.shape}')
        self._R = val
