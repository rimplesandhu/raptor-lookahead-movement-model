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
        object_id: str | int = 0
    ):
        super().__init__(nx, ny, dt, object_id)
        self.name = 'KF'

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        super().validate()
        if self.H is None:
            self.raiseit('KF: Need to initiate H matrix!')
        if self.F is None:
            self.raiseit('KF: Need to initiate F matrix!')

    def forecast(self) -> None:
        """Kalman filter forecast step"""
        super().forecast()
        self._m = self.qbar + self.F @ self.m
        self._P = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        self._store_this_step()

    def update(self) -> None:
        """Kalman filter update step"""
        y_pred = self.H @ self.m
        Smat = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        y_res = self.y_subtract(self.obs, y_pred)
        Kmat = self.P @ self.H.T @ Smat_inv
        x_res = Kmat @ y_res
        self._m = self.x_add(self.m, x_res)
        Tmat = np.eye(self.nx) - Kmat @ self.H
        self._P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T  # Joseph
        # self._P = (np.eye(self.nx) - Kmat @ self.H) @ self.P
        self._P = self.symmetrize(self.P) + np.diag([self.epsilon] * self.nx)
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
        self._store_this_step(update=True)

    def _backward_filter(self, smean_next, scov_next):
        """Backward filter"""
        mhat = self.F @ self.m
        Phat = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        Dmat = self.P @ self.F.T @ np.linalg.pinv(Phat, hermitian=True)
        self._m = self.x_add(self.m, Dmat @ self.x_subtract(smean_next, mhat))
        self._P += Dmat @ (scov_next - Phat) @ Dmat.T
        self._P = self.symmetrize(self.P) + np.diag([self.epsilon] * self.nx)
        if self.obs is not None:
            y_res = self.y_subtract(self.obs, self.H @ self.m)
            #Smat = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
            Smat = self.H @ Phat @ self.H.T + self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            #Kmat = self.P @ self.H.T @ Smat_inv
            Kmat = Phat @ self.H.T @ Smat_inv
            x_res = np.dot(Kmat, y_res)
            self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
        else:
            self._nees = np.nan
            self._nis = np.nan
            self._loglik = np.nan
