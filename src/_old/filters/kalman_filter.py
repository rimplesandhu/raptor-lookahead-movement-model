"""Kalman filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
import numpy as np
from .kalman_filter_base import KalmanFilterBase


class KalmanFilter(KalmanFilterBase):
    """Kalman Filter"""

    def initiate(self, *args, **kwargs):
        """Initiate function"""
        super().initiate(*args, **kwargs)
        if self.mat_H is None:
            self.raiseit('Need to initiate H matrix!')
        if self.mat_F is None:
            self.raiseit('Need to initiate F matrix!')

    def forecast(self) -> None:
        """Kalman filter forecast step"""
        super().forecast()
        self.m = self.vec_qbar + self.mat_F @ self.m
        self.P = self.mat_F @ self.P @ self.mat_F.T + \
            self.mat_G @ self.mat_Q @ self.mat_G.T
        self.store_this_timestep()

    def update(self) -> None:
        """Kalman filter update step"""
        yhat = self.mat_H @ self.m
        Smat = self.mat_H @ self.P @ self.mat_H.T + \
            self.mat_J @ self.R @ self.mat_J.T
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        yres = self.fun_subtract_y(self.y, yhat)
        Kmat = self.P @ self.mat_H.T @ Smat_inv
        xres = Kmat @ yres
        self._m = self.fun_subtract_x(self.m, -xres)
        Tmat = np.eye(self.nx) - Kmat @ self.mat_H
        self.P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T  # Joseph
        # self._P = (np.eye(self.nx) - Kmat @ self.mat_H) @ self.P
        self.P = self.symmetrize(self.P)
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        self.cur_metrics = self.compute_metrics(
            xres, Pmat_inv, yres, Smat_inv)
        self.store_this_timestep(update=True)

    def _backward_filter(self, smean_next, scov_next):
        """Backward filter"""
        Phat = self.mat_F @ self.P @ self.mat_F.T + \
            self.mat_G @ self.mat_Q @ self.mat_G.T
        Dmat = self.P @ self.mat_F.T @ np.linalg.pinv(Phat, hermitian=True)
        mhat = self.mat_F @ self.m
        xres = Dmat @ self.fun_subtract_x(smean_next, mhat)
        self.m = self.fun_subtract_x(self.m, -xres)
        self.P += Dmat @ (scov_next - Phat) @ Dmat.T
        self.P = self.symmetrize(self.P)
        self.cur_smetrics = self.compute_metrics()
        if self.y is not None:
            yhat = self.mat_H @ self.m
            yres = self.fun_subtract_y(self.y, yhat)
            Smat = self.mat_H @ Phat @ self.mat_H.T + \
                self.mat_J @ self.R @ self.mat_J.T
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Phat @ self.mat_H.T @ Smat_inv
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            xres = np.dot(Kmat, yres)
            self.cur_smetrics = self.compute_metrics(
                xres, Pmat_inv, yres, Smat_inv)
