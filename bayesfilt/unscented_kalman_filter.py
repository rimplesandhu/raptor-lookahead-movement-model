"""Unscented Kalman Filter class"""
from collections.abc import Callable
import numpy as np
from .kalman_filter_base import KalmanFilterBase
from .unscented_transform import UnscentedTransform


class UnscentedKalmanFilter(KalmanFilterBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    """Unscented Kalman Filter class"""

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0,
        dt_tol: float = 0.001,
        **kwargs
    ):
        super().__init__(nx, ny, dt, object_id, dt_tol)
        self.name = 'UKF'
        self.ut = UnscentedTransform(dim=nx, **kwargs)
        self.G = np.eye(self.nx)
        self.J = np.eye(self.ny)
        self.subtract_func: Callable | None = None

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        super().validate()
        if self.f is None:
            self.raiseit('Need to define dynamics function f()')
        if self.h is None:
            self.raiseit('Need to define observation function h()')
        if self.Q is None:
            self.raiseit('Need to initiate Q matrix!')

    def forecast(self) -> None:
        """UKF forecast step"""
        super().forecast()
        self._m, self._P, _ = self.ut.transform(self.m, self.P,
                                                self.f, self.subtract_func)
        self._P += self.Q
        self._store_this_step()

    def update(self) -> None:
        """UKF update step"""
        #self._P = self.symmetrize(self.P) + np.diag([1e-08] * self.nx)
        if np.any(np.isnan(self.P)) or np.any(self.P.diagonal() < 0.):
            print('\np forecast went wrong!')
            print(self.P.diagonal())
        y_pred, self._S, Pxy = self.ut.transform(self.m, self.P,
                                                 self.h)
        y_res = self.obs - y_pred
        self._S += self.R
        S_inv = np.linalg.pinv(self.S, hermitian=True)
        self._nis = np.linalg.multi_dot([y_res.T, S_inv, y_res])
        self._loglik = -0.5 * (self._ny * np.log(2. * np.pi) +
                               np.log(np.linalg.det(self.S)) + self.nis)
        self._K = Pxy @ S_inv
        x_res = self.K @ y_res
        self._m += x_res
        self._P -= self.K @ self.S @ self.K.T
        if np.any(np.isnan(self.P)) or np.any(self.P.diagonal() < 0.):
            print('\nupdate went wrong!')
            print(self.P.diagonal())
            inds = np.where(self.P.diagonal() < 0.)[0]
            for ind in inds:
                self._P[ind, ind] = 1e-2
        P_inv = np.linalg.pinv(self.P, hermitian=True)
        if self.truth is not None:
            x_res = self.m - self.truth
        self._nees = np.linalg.multi_dot([x_res.T, P_inv, x_res])
        self._store_this_step(update=True)

    def backward_update(self):
        """UKS backward update"""
        smean_next = self.history_smoother['state_mean'][-1].copy()
        scov_next = self.history_smoother['state_cov'][-1].copy()
        mhat, Phat, Cmat = self.ut.transform(self.m, self.P, self.f)
        Phat += self.G @ self.Q @ self.G.T
        Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True)
        self._m += Dmat @ (smean_next - mhat)
        self._P += Dmat @ (scov_next - Phat) @ Dmat.T
        if self.obs is not None:
            y_pred, self._S, Pxy = self.ut.transform(self.m, self.P, self.h)
            yres = self.obs - y_pred
            self._S += self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(self.S, hermitian=True)
            self._nis = np.linalg.multi_dot([yres.T, Smat_inv, yres])
            self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) +
                                   np.log(np.linalg.det(self.S)) + self.nis)
            if self.truth is not None:
                xres = self.m - self.truth
            else:
                Kmat = Pxy @ Smat_inv
                xres = np.dot(Kmat, yres)
            scov_inv = np.linalg.pinv(self.P, hermitian=True)
            self._nees = np.linalg.multi_dot([xres.T, scov_inv, xres])
