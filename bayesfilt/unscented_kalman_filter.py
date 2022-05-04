"""Unscented Kalman Filter class"""
from collections.abc import Sequence
import numpy as np
from numpy import ndarray
from .kalman_filter_base import KalmanFilterBase
from .unscented_transform import UnscentedTransform


class UnscentedKalmanFilter(KalmanFilterBase):
    # pylint: disable=invalid-name
    """Unscented Kalman Filter class"""

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0,
        **kwargs
    ):
        super().__init__('UKF', nx, ny, dt, object_id)
        self.ut = UnscentedTransform(dim=nx, **kwargs)

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        if (self._P is None) or (self._m is None):
            self._raiseit('Need to initiate state, use initiate_state()')
        if self._h is None:
            self._raiseit('Need to initiate measurement equation h()!')
        if self._f is None:
            self._raiseit('Need to initiate dynamics equation f()!')
        if self._R is None:
            self._raiseit('Need to initiate R matrix!')
        if self._Q is None:
            self._raiseit('Need to initiate Q matrix!')

    def forecast(self) -> None:
        """Unscented Kalman filter forecast step"""
        self._time_elapsed += self.dt
        if self._m is None:
            self._raiseit('Need to initiate state first!')
        self._m, self._P, _ = self.ut.transform(self._m, self._P, self._f)
        self._P += self._Q
        self._nis = np.nan
        self._nees = np.nan
        self._loglik = np.nan
        self._store_this_step()

    def update(self, y_obs: ndarray) -> None:
        """Kalman filter update step"""
        y_obs = self.mat_setter(y_obs, (self._ny, ))
        y_pred, self._S, Pxy = self.ut.transform(self._m, self._P, self._h)
        y_res = y_obs - y_pred
        self._S += self._R
        S_inv = np.linalg.pinv(self._S, hermitian=True)
        self._nis = np.linalg.multi_dot([y_res.T, S_inv, y_res])
        self._loglik = -0.5 * (self._ny * np.log(2. * np.pi) +
                               np.log(np.linalg.det(self._S)) + self._nis)
        self._K = Pxy @ S_inv
        x_res = np.dot(self._K, y_res)
        self._m += x_res
        self._P -= self._K @ self._S @ self._K.T
        P_inv = np.linalg.pinv(self._P, hermitian=True)
        self._nees = np.linalg.multi_dot([x_res.T, P_inv, x_res])
        self._store_this_step(y_obs)

    def filter(
        self,
        tvec: ndarray,
        list_of_yobs: Sequence[ndarray],
        list_of_rmat: Sequence[ndarray] | None = None,
        dt_tol: float = 0.001
    ) -> None:
        """Run filtering assuming F, H, Q matrices are time invariant"""
        self.validate()
        tvec = np.asarray(tvec)
        assert tvec.ndim == 1, f'Wrong time series shape {tvec.shape}'
        assert len(list_of_yobs) == tvec.size, 'Size mismatch: time vs yvec'
        if list_of_rmat is not None:
            assert len(list_of_rmat) == tvec.size, 'Size mismatch:time vs rmat'
        k = 0
        while k < tvec.size:
            if self._time_elapsed - tvec[k] > dt_tol:
                self._raiseit(f'Skipping observation, lower the dt={self.dt}!')
            if abs(self._time_elapsed - tvec[k]) < dt_tol:
                if list_of_rmat is not None:
                    assert len(
                        list_of_rmat) == tvec.size, 'Size mismatch: time vs rmat'
                    self._R = list_of_rmat[k]
                self.update(list_of_yobs[k])
                k += 1
            else:
                self.forecast()
        # for k, v in self.metrics.items():
        #     print(f'Filter: {k} = {np.around(v,3)}')

    def smoother(self):
        # pylint: disable=too-many-locals
        """Run smoothing assuming model/measurement eq are time invariant"""
        nsteps = len(self._history['time'])
        if nsteps < 1:
            self._raiseit('No filter history found, run filter first!')
        self._erase_history_smoother()
        for i in reversed(range(nsteps)):
            time = self._history['time'][i]
            if i == nsteps - 1:
                smean = self._history['state_mean'][i]
                scov = self._history['state_cov'][i]
            else:
                umean = self._history['state_mean'][i]
                ucov = self._history['state_cov'][i]
                fcov = self._F @ ucov @ self._F.T + self._Q
                fcov_inv = np.linalg.pinv(fcov, hermitian=True)
                gmat = ucov @ self._F.T @ fcov_inv
                smean = umean + gmat @ (smean - self._F @ umean)
                scov = ucov + gmat @ (scov - fcov) @ gmat.T
            yobs = self._history['obs'][i]
            rmat = self._history['rmat'][i]
            nis = np.nan
            nees = np.nan
            if yobs is not None:
                yres = yobs - self._H @ smean
                Smat = self._H @ scov @ self._H.T + rmat
                Smat_inv = np.linalg.pinv(Smat, hermitian=True)
                nis = np.linalg.multi_dot([yres.T, Smat_inv, yres])
                loglik = -0.5 * (self._ny * np.log(2. * np.pi) +
                                 np.log(np.linalg.det(Smat)) + nis)
                Kmat = scov @ self._H.T @ Smat_inv
                xres = np.dot(Kmat, yres)
                scov_inv = np.linalg.pinv(scov, hermitian=True)
                nees = np.linalg.multi_dot([xres.T, scov_inv, xres])
            self._history_smoother['time'].append(time)
            self._history_smoother['obs'].append(yobs)
            self._history_smoother['state_mean'].append(smean)
            self._history_smoother['state_cov'].append(scov)
            self._history_smoother['nees'].append(nees)
            self._history_smoother['nis'].append(nis)
            self._history_smoother['loglik'].append(loglik)
            self._history_smoother['rmat'].append(rmat)
        for v in self._history_smoother.values():
            v.reverse()
        # for k, v in self.metrics_smoother.items():
        #     print(f'Smoother: {k} = {np.around(v,3)}')
