"""Kalman filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from ._base_filter import KalmanFilterBase
from ._variables import FilterVariables
from .utils import check_mat, symmetrize_mat

from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass(kw_only=True)
class KalmanFilter(KalmanFilterBase):
    """Kalman Filter"""

    mat_F: ndarray = field(repr=False)
    mat_H: ndarray = field(repr=False)
    mat_Q: ndarray = field(repr=False)

    def __post_init__(self):
        """post initiation function"""
        # validate user input
        super().__post_init__()
        self.mat_F = check_mat(self.mat_F, (self.nx, self.nx))
        self.mat_H = check_mat(self.mat_H, (self.ny, self.nx))
        self.mat_Q = check_mat(self.mat_Q, (self.nx, self.nx))

    def forecast_step(
        self,
        out_vars: FilterVariables
    ) -> None:
        """Kalman filter forecast step"""
        # update mean and cov
        out_vars.m = self.mat_F @ self.vars.m
        out_vars.P = self.mat_F @ self.vars.P @ self.mat_F.T + self.mat_Q

    def update_step(
        self,
        out_vars: FilterVariables
    ) -> None:
        """Kalman filter update step"""
        # residual in obs
        yhat = self.mat_H @ self.vars.m
        Smat = self.mat_H @ self.vars.P @ self.mat_H.T + out_vars.R
        out_vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
        out_vars.yres = self.fun_subtract_y(out_vars.y, yhat)

        # Updated state mean
        Kmat = self.vars.P @ self.mat_H.T @ out_vars.Sinv
        out_vars.mres = Kmat @ out_vars.yres
        out_vars.m = self.fun_subtract_x(self.vars.m, -out_vars.mres)

        # update state cov
        Tmat = np.eye(self.nx) - Kmat @ self.mat_H
        out_vars.P = Tmat @ self.vars.P @ Tmat.T
        out_vars.P += Kmat @ out_vars.R @ Kmat.T  # Joseph
        # self.vars.P = (np.eye(self.nx) - Kmat @ self.mat_H) @ self.vars.P
        out_vars.P = symmetrize_mat(out_vars.P, eps=self.epsilon)
        out_vars.Pinv = np.linalg.pinv(out_vars.P, hermitian=True)

    def backward_update(
        self,
        m_next: ndarray,
        P_next: ndarray
    ):
        """Backward filter for smoothing"""

        # predicted covaraince matrix
        Phat = self.mat_F @ self.vars.P @ self.mat_F.T + self.mat_Q
        Phat_inv = np.linalg.pinv(Phat, hermitian=True)

        # smoothed mean
        Dmat = self.vars.P @ self.mat_F.T @ Phat_inv
        mhat = self.mat_F @ self.vars.m
        self.vars.mres = Dmat @ self.fun_subtract_x(m_next, mhat)
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)

        # smoother covariance
        self.vars.P += Dmat @ (P_next - Phat) @ Dmat.T
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)
        self.vars.Pinv = np.linalg.pinv(self.vars.P, hermitian=True)

        # if data is encountered
        if self.vars.y is not None:
            yhat = self.mat_H @ self.vars.m
            self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)
            Smat = self.mat_H @ Phat @ self.mat_H.T + self.vars.R
            self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Phat @ self.mat_H.T @ self.vars.Sinv
            self.vars.mres = np.dot(Kmat, self.vars.yres)
