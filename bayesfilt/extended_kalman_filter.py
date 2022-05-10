"""Extended Kalman filter class"""
import numpy as np
from .kalman_filter_base import KalmanFilterBase


class ExtendedKalmanFilter(KalmanFilterBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    # pylint: disable=not-callable
    """Extended Kalman Filter"""

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0
    ):
        super().__init__(nx, ny, dt, object_id)
        self.name = 'EKF'

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        super().validate()
        if self.f is None:
            self.raiseit('Need to define dynamics function f()')
        if self.h is None:
            self.raiseit('Need to define observation function h()')
        if self.compute_F is None:
            self.raiseit('Need to define function compute_F to compute F!')
        if self.compute_Q is None:
            self.raiseit('Need to define function compute_Q to compute Q!')
        if self.compute_H is None:
            self.raiseit('Need to define function compute_H to compute H!')

    def forecast(self) -> None:
        """Kalman filter forecast step"""
        super().forecast()
        self.F = self.compute_F(self.m, self.qbar)
        self.Q = self.compute_Q(self.m, self.qbar)
        if self.compute_G is not None:
            self.G = self.compute_G(self.m, self.qbar)
        self._m = self.f(self.m, self.qbar)
        self._P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        self._P = self.symmetrize(self.P)
        self._store_this_step()

    def update(self) -> None:
        """Kalman filter update step"""
        self.H = self.compute_H(self.m, self.rbar)
        if self.compute_J is not None:
            self.J = self.compute_J(self.m, self.rbar)
        y_pred = self.h(self.m, self.rbar)
        Smat = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        y_res = self.residual(self.obs, y_pred)
        Kmat = self.P @ self.H.T @ Smat_inv
        x_res = Kmat @ y_res
        self._m += x_res
        Tmat = np.eye(self.nx) - Kmat @ self.H  # Joseph's form, num stable
        self._P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T
        self._P = self.symmetrize(self.P) + np.diag([self.epsilon] * self.nx)
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
        self._store_this_step(update=True)

    def _backward_filter(self):
        """Backward filter"""
        smean_next = self.history['smoother_mean'][-1]
        scov_next = self.history['smoother_cov'][-1]
        self.F = self.compute_F(self.m, self.qbar)
        if self.compute_G is not None:
            self.G = self.compute_G(self.m, self.qbar)
        self.Q = self.compute_Q(self.m, self.qbar)
        fcov = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        fcov_inv = np.linalg.pinv(fcov, hermitian=True)
        gmat = self.P @ self.F.T @ fcov_inv
        self._m += gmat @ (smean_next - self.f(self.m, self.qbar))
        self._P += gmat @ (scov_next - fcov) @ gmat.T
        if self.obs is not None:
            self.H = self.compute_H(self.m, self.rbar)
            if self.compute_J is not None:
                self.J = self.compute_J(self.m, self.rbar)
            y_pred = self.h(self.m, self.rbar)
            y_res = self.residual(self.obs, y_pred)
            Smat = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = self.P @ self.H.T @ Smat_inv
            x_res = np.dot(Kmat, y_res)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
