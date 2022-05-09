"""Kalman filter class"""
import numpy as np
from .kalman_filter_base import KalmanFilterBase


class KalmanFilter(KalmanFilterBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    """Kalman Filter"""

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0,
        dt_tol: float = 0.001
    ):
        super().__init__(nx, ny, dt, object_id, dt_tol)
        self.name = 'KF'
        self.G = np.eye(self.nx)
        self.J = np.eye(self.ny)

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        super().validate()
        if self.H is None:
            self.raiseit('Need to initiate H matrix!')
        if self.F is None:
            self.raiseit('Need to initiate F matrix!')
        if self.Q is None:
            self.raiseit('Need to initiate Q matrix!')

    def forecast(self) -> None:
        """Kalman filter forecast step"""
        super().forecast()
        self._m = self.F @ self.m
        self._P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        self._store_this_step()

    def update(self) -> None:
        """Kalman filter update step"""
        y_pred = self.H @ self.m
        self._S = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
        S_inv = np.linalg.pinv(self.S, hermitian=True)
        y_res = self.obs - y_pred
        self._K = self.P @ self.H.T @ S_inv
        x_res = self.K @ y_res
        self._m += x_res
        self._P = (np.eye(self.nx) - self.K @ self.H) @ self.P
        self._nis = np.linalg.multi_dot([y_res.T, S_inv, y_res])
        self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) +
                               np.log(np.linalg.det(self.S)) + self.nis)
        P_inv = np.linalg.pinv(self.P, hermitian=True)
        if self.truth is not None:
            x_res = self.m - self.truth
        self._nees = np.linalg.multi_dot([x_res.T, P_inv, x_res])
        self._store_this_step(update=True)

    def backward_update(self):
        """Backward filter"""
        smean_next = self.history_smoother['state_mean'][-1].copy()
        scov_next = self.history_smoother['state_cov'][-1].copy()
        fcov = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        fcov_inv = np.linalg.pinv(fcov, hermitian=True)
        gmat = self.P @ self.F.T @ fcov_inv
        self._m += gmat @ (smean_next - self.F @ self.m)
        self._P += gmat @ (scov_next - fcov) @ gmat.T
        if self.obs is not None:
            yres = self.obs - self.H @ self.m
            self._S = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(self.S, hermitian=True)
            self._nis = np.linalg.multi_dot([yres.T, Smat_inv, yres])
            self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) +
                                   np.log(np.linalg.det(self.S)) + self.nis)
            if self.truth is not None:
                xres = self.m - self.truth
            else:
                Kmat = self.P @ self.H.T @ Smat_inv
                xres = np.dot(Kmat, yres)
            scov_inv = np.linalg.pinv(self.P, hermitian=True)
            self._nees = np.linalg.multi_dot([xres.T, scov_inv, xres])
