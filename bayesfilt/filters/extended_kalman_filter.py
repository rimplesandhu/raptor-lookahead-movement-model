"""Extended Kalman filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from ._base_filter import KalmanFilterBase
from ._variables import FilterVariables
from .utils import Func1to1, symmetrize_mat, check_mat

from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray


@dataclass(kw_only=True)
class ExtendedKalmanFilter(KalmanFilterBase):
    """Extended Kalman Filter"""
    # nonlinear model dynamics
    fun_f: Func1to1 = field(repr=False)
    fun_Fjac: Func1to1 = field(repr=False)

    # Nonlinear model error
    fun_Q: Func1to1 = field(repr=False)

    # Nonnlinear observation eq
    fun_h: Func1to1 = field(repr=False)
    fun_Hjac: Func1to1 = field(repr=False)

    def __post_init__(self):
        """post initiation function"""
        # validate user input
        super().__post_init__()
        _ = check_mat(self.fun_f(np.zeros(self.nx)), self.nx)
        _ = check_mat(self.fun_Fjac(np.zeros(self.nx)), (self.nx, self.nx))
        _ = check_mat(self.fun_h(np.zeros(self.nx)), self.ny)
        _ = check_mat(self.fun_Hjac(np.zeros(self.nx)), (self.ny, self.nx))
        _ = check_mat(self.fun_Q(np.zeros(self.nx)), (self.nx, self.nx))

    def forecast_step(
        self,
        out_vars: FilterVariables
    ) -> None:
        """EKF forecast step"""
        # compute jacobians
        Fmat = self.fun_Fjac(self.vars.m)
        Qmat = self.fun_Q(self.vars.m)

        # forecast step
        out_vars.m = self.fun_f(self.vars.m)
        out_vars.P = Fmat @ self.vars.P @ Fmat.T + Qmat
        out_vars.P = symmetrize_mat(out_vars.P, eps=self.epsilon)

    def update_step(
        self,
        out_vars: FilterVariables
    ) -> None:
        """EKF update step"""

        # Observation equation/jacobians
        Hmat = self.fun_Hjac(self.vars.m)
        yhat = self.fun_h(self.vars.m)

        # observation residual and precision
        Smat = Hmat @ self.vars.P @ Hmat.T + out_vars.R
        out_vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
        out_vars.yres = self.fun_subtract_y(out_vars.y, yhat)

        # innovation and updated mean
        Kmat = self.vars.P @ Hmat.T @ out_vars.Sinv
        out_vars.mres = Kmat @ out_vars.yres
        out_vars.m = self.fun_subtract_x(self.vars.m, -out_vars.mres)

        # Joseph's form, num stable for computing updated state cov
        Tmat = np.eye(self.nx) - Kmat @ Hmat
        out_vars.P = Tmat @ self.vars.P @ Tmat.T + Kmat @ out_vars.R @ Kmat.T
        out_vars.P = symmetrize_mat(out_vars.P, eps=self.epsilon)
        out_vars.Pinv = np.linalg.pinv(out_vars.P, hermitian=True)

    def backward_update(
        self,
        m_next: ndarray,
        P_next: ndarray
    ):
        """Backward filter for smoothing"""
        # dynamics eqn
        Fmat = self.fun_Fjac(self.vars.m)
        Qmat = self.fun_Q(self.vars.m)

        # predicted covaraince matrix
        Phat = Fmat @ self.vars.P @ Fmat.T + Qmat
        Phat_inv = np.linalg.pinv(Phat, hermitian=True)

        # smoothed mean
        Dmat = self.vars.P @ Fmat.T @ Phat_inv
        mhat = self.fun_f(self.vars.m)
        self.vars.mres = Dmat @ self.fun_subtract_x(m_next, mhat)
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)

        # smoother covaraince
        self.vars.P += Dmat @ (P_next - Phat) @ Dmat.T
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)
        self.vars.Pinv = np.linalg.pinv(self.vars.P, hermitian=True)

        # if data is encountered
        if self.vars.y is not None:
            Hmat = self.fun_Hjac(self.vars.m)
            yhat = self.fun_h(self.vars.m)
            self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)
            Smat = Hmat @ Phat @ Hmat.T + self.vars.R
            self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Phat @ Hmat.T @ self.vars.Sinv
            self.vars.mres = np.dot(Kmat, self.vars.yres)
