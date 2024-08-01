"""Kalman filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from ._filter_base import KalmanFilterBase
from .utils import assign_mat, symmetrize_mat

from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass(kw_only=True)
class KalmanFilter(KalmanFilterBase):
    """Kalman Filter"""

    mat_F: ndarray = field(repr=False)
    mat_H: ndarray = field(repr=False)
    mat_Q: ndarray = field(repr=False)
    mat_G: ndarray | None = field(default=None, repr=False)
    mat_J: ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        """Post initialization"""
        super().__post_init__()
        if self.mat_G is None:
            self.printit('Setting mat_G as identity matrix!')
            self.mat_G = np.eye(self.nx)
        if self.mat_J is None:
            self.printit('Setting mat_J as identity matrix!')
            self.mat_J = np.eye(self.ny)

    def forecast(self, flag: str | None = None) -> None:
        """Kalman filter forecast step"""
        # update mean and cov
        self.vars.flag = flag
        self.vars.m = self.mat_F @ self.vars.m
        self.vars.P = self.mat_F @ self.vars.P @ self.mat_F.T
        self.vars.P += self.mat_G @ self.mat_Q @ self.mat_G.T

        # run postproeccesing steps
        self.forecast_postprocess()

    def update(
        self,
        obs_y: ndarray,
        obs_R: ndarray,
        obs_flag: str | None = None
    ) -> None:
        """Kalman filter update step"""
        # update observation vars in tracker
        self.vars.y = assign_mat(obs_y, self.ny)
        self.vars.R = assign_mat(obs_R, (self.ny, self.ny))

        # residual in obs
        yhat = self.mat_H @ self.vars.m
        Smat = self.mat_H @ self.vars.P @ self.mat_H.T
        Smat += self.mat_J @ self.vars.R @ self.mat_J.T
        self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
        self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)

        # Updated state mean
        Kmat = self.vars.P @ self.mat_H.T @ self.vars.Sinv
        self.vars.mres = Kmat @ self.vars.yres
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)

        # update state cov
        Tmat = np.eye(self.nx) - Kmat @ self.mat_H
        self.vars.P = Tmat @ self.vars.P @ Tmat.T
        self.vars.P += Kmat @ self.vars.R @ Kmat.T  # Joseph
        # self.vars.P = (np.eye(self.nx) - Kmat @ self.mat_H) @ self.vars.P
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)
        self.vars.Pinv = np.linalg.pinv(self.vars.P, hermitian=True)

        # postprocessing
        self.vars.flag = obs_flag
        self.update_postprocess()

    def backward_update(
        self,
        m_next: ndarray,
        P_next: ndarray
    ):
        """Backward filter for smoothing"""
        Phat = self.mat_F @ self.vars.P @ self.mat_F.T
        Phat += self.mat_G @ self.mat_Q @ self.mat_G.T
        Phat_inv = np.linalg.pinv(Phat, hermitian=True)
        Dmat = self.vars.P @ self.mat_F.T @ Phat_inv
        mhat = self.mat_F @ self.vars.m
        self.vars.mres = Dmat @ self.fun_subtract_x(m_next, mhat)
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)
        self.vars.P += Dmat @ (P_next - Phat) @ Dmat.T
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)

        # if data is encountered
        if self.vars.y is not None:
            yhat = self.mat_H @ self.vars.m
            self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)
            Smat = self.mat_H @ Phat @ self.mat_H.T
            Smat += self.mat_J @ self.vars.R @ self.mat_J.T
            self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Phat @ self.mat_H.T @ self.vars.Sinv
            self.vars.Pinv = np.linalg.pinv(self.vars.P, hermitian=True)
            self.vars.mres = np.dot(Kmat, self.vars.yres)
