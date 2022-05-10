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
        super().__init__(nx, ny, dt, object_id)
        self.name = 'UKF'
        self.ut = UnscentedTransform(dim=nx, **kwargs)

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
        self._m, self._P, _ = self.ut.transform(self.m, self.P, self.f)
        self._P += self.Q
        self._store_this_step()

    def update(self) -> None:
        """UKF update step"""
        y_pred, Smat, Pxy = self.ut.transform(self.m, self.P, self.h)
        y_res = self.residual(self.obs, y_pred)
        Smat += self.R
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        Kmat = Pxy @ Smat_inv
        x_res = Kmat @ y_res
        self._m += x_res
        self._P -= Kmat @ Smat @ Kmat.T
        self._P = self.symmetrize(self.P) + np.diag([self.epsilon] * self.nx)
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
        self._store_this_step(update=True)

    def _backward_filter(self):
        """Backward filter"""
        smean_next = self.history['smoother_mean'][-1]
        scov_next = self.history['smoother_cov'][-1]
        mhat, Phat, Cmat = self.ut.transform(self.m, self.P, self.f)
        Phat += self.G @ self.Q @ self.G.T
        Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True)
        self._m += Dmat @ (smean_next - mhat)
        self._P += Dmat @ (scov_next - Phat) @ Dmat.T
        if self.obs is not None:
            y_pred, Smat, Pxy = self.ut.transform(self.m, self.P, self.h)
            y_res = self.residual(self.obs, y_pred)
            Smat += self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            Kmat = Pxy @ Smat_inv
            x_res = np.dot(Kmat, y_res)
            self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
