"""Extended Kalman filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from ._filter_base import KalmanFilterBase
from .utils import Func1to1, symmetrize_mat, assign_mat

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
    fun_Gjac: Func1to1 | None = field(default=None, repr=False)

    # Nonnlinear observation eq
    fun_h: Func1to1 = field(repr=False)
    fun_Hjac: Func1to1 = field(repr=False)

    # Nonlinear observation error
    fun_Jjac: Func1to1 | None = field(default=None, repr=False)

    def __post_init__(self):
        """Post initialization"""
        super().__post_init__()

        # check if fun_Gjac supplied
        if self.fun_Gjac is None:
            self.printit(f'Setting fun_Gjac = Identity({self.nx}) !')
            self.fun_Gjac = lambda x: np.eye(self.nx)

        # check if fun_Jjac supplied
        if self.fun_Jjac is None:
            self.printit(f'Setting fun_Jjac = Identity({self.ny}) !')
            self.fun_Jjac = lambda x: np.eye(self.ny)

    def forecast(self, flag: str | None = None) -> None:
        """EKF forecast step"""

        # compute jacobians
        Fmat = self.fun_Fjac(self.vars.m)
        Qmat = self.fun_Q(self.vars.m)
        Gmat = self.fun_Gjac(self.vars.m)

        # forecast step
        self.vars.flag = flag
        self.vars.m = self.fun_f(self.vars.m)
        self.vars.P = Fmat @ self.vars.P @ Fmat.T + Gmat @ Qmat @ Gmat.T
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)

        # finish
        self.forecast_postprocess()

    def update(
        self,
        obs_y: ndarray,
        obs_R: ndarray,
        obs_flag: str | None = None
    ) -> None:
        """EKF update step"""
        # update observation vars
        self.vars.y = assign_mat(obs_y, self.ny)
        self.vars.R = assign_mat(obs_R, (self.ny, self.ny))

        # Observation equation/jacobians
        Hmat = self.fun_Hjac(self.vars.m)
        Jmat = self.fun_Jjac(self.vars.m)
        yhat = self.fun_h(self.vars.m)

        # observation residual and precision
        Smat = Hmat @ self.vars.P @ Hmat.T + Jmat @ self.vars.R @ Jmat.T
        self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
        self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)

        # innovation and updated mean
        Kmat = self.vars.P @ Hmat.T @ self.vars.Sinv
        self.vars.mres = Kmat @ self.vars.yres
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)

        # Joseph's form, num stable for computing updated state cov
        Tmat = np.eye(self.nx) - Kmat @ Hmat
        self.vars.P = Tmat @ self.vars.P @ Tmat.T + Kmat @ self.vars.R @ Kmat.T
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
        # dynamics eqn
        Fmat = self.fun_Fjac(self.vars.m)
        Gmat = self.fun_Gjac(self.vars.m)
        Qmat = self.fun_Q(self.vars.m)

        # update mean and cov
        Phat = Fmat @ self.vars.P @ Fmat.T
        Phat += Gmat @ Qmat @ Gmat.T
        Phat_inv = np.linalg.pinv(Phat, hermitian=True)
        Dmat = self.vars.P @ Fmat.T @ Phat_inv
        mhat = self.fun_f(self.vars.m)
        self.vars.mres = Dmat @ self.fun_subtract_x(m_next, mhat)
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)
        self.vars.P += Dmat @ (P_next - Phat) @ Dmat.T
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)

        # if data is encountered
        if self.vars.y is not None:
            Hmat = self.fun_Hjac(self.vars.m)
            Jmat = self.fun_Jjac(self.vars.m)
            yhat = self.fun_h(self.vars.m)
            self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)
            Smat = Hmat @ Phat @ Hmat.T
            Smat += Jmat @ self.vars.R @ Jmat.T
            self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Phat @ Hmat.T @ self.vars.Sinv
            self.vars.Pinv = np.linalg.pinv(self.vars.P, hermitian=True)
            self.vars.mres = np.dot(Kmat, self.vars.yres)
