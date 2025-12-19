"""Extended Kalman filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from functools import partial
import numpy as np
from .kalman_filter_base import KalmanFilterBase


class ExtendedKalmanFilter(KalmanFilterBase):
    """Extended Kalman Filter"""

    def initiate(self, *args, **kwargs):
        """Initiate function"""
        super().initiate(*args, **kwargs)
        if self.fun_f is not None:
            if self.mat_F is not None:
                object.__setattr__(
                    self, 'fun_Fjac', partial(self.v2m, self.mat_F))
            else:
                self.raiseit('Need to define either mat_F or fun_Fjac')
        if self.fun_h is not None:
            if self.mat_H is not None:
                object.__setattr__(
                    self, 'fun_Hjac', partial(self.v2m, self.mat_H))
            else:
                self.raiseit('Need to define either mat_H or fun_Hjac')

    def forecast(self) -> None:
        """EKF forecast step"""
        super().forecast()
        Fmat = self.fun_Fjac(self.m, self.vec_qbar)
        Qmat = self.fun_Q(self.m, self.vec_qbar)
        Gmat = self.fun_Gjac(self.m, self.vec_qbar)
        self.m = self.fun_f(self.m, self.vec_qbar)
        self.P = Fmat @ self.P @ Fmat.T + Gmat @ Qmat @ Gmat.T
        self.P = self.symmetrize(self.P)
        self.store_this_timestep()

    def update(self) -> None:
        """EKF update step"""
        Hmat = self.fun_Hjac(self.m, self.vec_rbar)
        Jmat = self.fun_Jjac(self.m, self.vec_rbar)
        yhat = self.fun_h(self.m, self.vec_rbar)
        Smat = Hmat @ self.P @ Hmat.T + Jmat @ self.R @ Jmat.T
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        yres = self.fun_subtract_y(self.y, yhat)
        Kmat = self.P @ Hmat.T @ Smat_inv
        xres = Kmat @ yres
        self.m = self.fun_subtract_x(self.m, -xres)
        Tmat = np.eye(self.nx) - Kmat @ Hmat  # Joseph's form, num stable
        self.P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T
        Pmat_inv = np.linalg.pinv(self.symmetrize(self.P), hermitian=True)
        self.cur_metrics = self.compute_metrics(
            xres, Pmat_inv, yres, Smat_inv)
        self.store_this_timestep(update=True)

    def _backward_filter(self, smean_next, scov_next):
        """Backward filter for smoother"""
        Fmat = self.fun_Fjac(self.m, self.vec_qbar)
        Gmat = self.fun_Gjac(self.m, self.vec_qbar)
        Qmat = self.fun_Q(self.m, self.vec_qbar)
        Phat = Fmat @ self.P @ Fmat.T + Gmat @ Qmat @ Gmat.T
        Dmat = self.P @ Fmat.T @ np.linalg.pinv(Phat, hermitian=True)
        mhat = self.fun_f(self.m, self.vec_qbar)
        xres = Dmat @ self.fun_subtract_x(smean_next, mhat)
        self.m = self.fun_subtract_x(self.m, -xres)
        self.P += Dmat @ (scov_next - Phat) @ Dmat.T
        self.P = self.symmetrize(self.P)
        self.cur_smetrics = self.compute_metrics()
        if self.y is not None:
            Hmat = self.fun_Hjac(self.m, self.vec_rbar)
            Jmat = self.fun_Jjac(self.m, self.vec_rbar)
            yhat = self.fun_h(self.m, self.vec_rbar)
            yres = self.fun_subtract_y(self.y, yhat)
            Smat = Hmat @ Phat @ Hmat.T + Jmat @ self.R @ Jmat.T
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Phat @ Hmat.T @ Smat_inv
            xres = np.dot(Kmat, yres)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            self.cur_smetrics = self.compute_metrics(
                xres, Pmat_inv, yres, Smat_inv)
