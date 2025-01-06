"""Unscented Kalman Filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from ._base_filter import KalmanFilterBase
from ._variables import FilterVariables
from .utils import Func1to1, symmetrize_mat
from .unscented_transform import UnscentedTransform

import numpy as np
from numpy import ndarray
from dataclasses import dataclass, field
from functools import partial


@dataclass(kw_only=True)
class UnscentedKalmanFilter(KalmanFilterBase):
    """Unscented Kalman Filter"""

    # nonlinear dynamics and obs equation
    fun_f: Func1to1 = field(repr=False)  # dynamics
    fun_h: Func1to1 = field(repr=False)  # observation eqn
    mat_Q: ndarray = field(repr=False)  # error cov martix

    # UKF parameters
    alpha: float | None = None
    beta: float | None = None
    kappa: float | None = None
    use_chol: bool | None = None

    # weighted mean functions (to handles angles)
    fun_weighted_mean_x: Func1to1 | None = field(default=None, repr=False)
    fun_weighted_mean_y: Func1to1 | None = field(default=None, repr=False)

    def __post_init__(self):
        """Post initialization"""
        super().__post_init__()

        # check UKF parameters
        if self.alpha is None:
            self.alpha = 1.
            self.printit(f'Setting alpha={self.alpha}')

        if self.kappa is None:
            self.kappa = 3-self.nx
            self.printit(f'Setting kappa={self.kappa}')

        if self.beta is None:
            self.beta = 0.
            self.printit(f'Setting beta={self.beta}')

        if self.use_chol is None:
            self.use_chol = False
            self.printit(f'Setting use_chol={self.use_chol}')

        if self.fun_weighted_mean_x is None:
            self.fun_weighted_mean_x = partial(np.average, axis=0)
            self.printit(f'Setting fun_weighted_mean_x=np.average')

        if self.fun_weighted_mean_y is None:
            self.fun_weighted_mean_y = partial(np.average, axis=0)
            self.printit(f'Setting fun_weighted_mean_y=np.average')

        self.ut_fun_f = UnscentedTransform(
            model_fun=self.fun_f,
            dim=self.nx,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            use_cholesky=self.use_chol,
            fun_subtract_in=self.fun_subtract_x,
            fun_subtract_out=self.fun_subtract_x,
            fun_weighted_mean_out=self.fun_weighted_mean_x
        )
        self.ut_fun_h = UnscentedTransform(
            dim=self.nx,
            alpha=self.alpha,
            beta=self.beta,
            kappa=self.kappa,
            use_cholesky=self.use_chol,
            fun_subtract_in=self.fun_subtract_x,
            model_fun=self.fun_h,
            fun_subtract_out=self.fun_subtract_y,
            fun_weighted_mean_out=self.fun_weighted_mean_y
        )

    def forecast_step(
        self,
        out_vars: FilterVariables
    ) -> None:
        """UKF forecast step"""
        _mvec, _Pmat, _ = self.ut_fun_f.transform(self.vars.m, self.vars.P)
        out_vars.m = _mvec
        out_vars.P = _Pmat
        out_vars.P += self.mat_Q
        out_vars.P = symmetrize_mat(out_vars.P, eps=self.epsilon)

    def update_step(
        self,
        out_vars: FilterVariables
    ) -> None:
        """UKF update step"""

        # observation equation through unscented tranform
        yhat, Smat, Pxy = self.ut_fun_h.transform(
            m=self.vars.m,
            P=self.vars.P
        )

        # observation residual and precision
        Smat += out_vars.R
        out_vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
        out_vars.yres = self.fun_subtract_y(out_vars.y, yhat)

        # innovation and updated mean
        Kmat = Pxy @ out_vars.Sinv
        out_vars.mres = Kmat @ out_vars.yres
        out_vars.m = self.fun_subtract_x(self.vars.m, -out_vars.mres)

        # Joseph's form, num stable for computing updated state cov
        out_vars.P = self.vars.P - Kmat @ Pxy.T
        out_vars.P = symmetrize_mat(out_vars.P, eps=self.epsilon)
        out_vars.Pinv = np.linalg.pinv(out_vars.P, hermitian=True)

        # Tmat = np.eye(self.nx) - Kmat @ Hmat
        # self.P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T  # Joseph
        # self._P -= Kmat @ Smat @ Kmat.T

    def backward_update(
        self,
        m_next: ndarray,
        P_next: ndarray
    ):
        """Backward filter for smoothing"""

        # pass through dynamics equation
        mhat, Phat, Cmat = self.ut_fun_f.transform(self.vars.m, self.vars.P)
        Phat += self.mat_Q
        Phat_inv = np.linalg.pinv(Phat, hermitian=True)

        # smoothed mean
        Dmat = Cmat @ Phat_inv
        self.vars.mres = Dmat @ self.fun_subtract_x(m_next, mhat)
        self.vars.m = self.fun_subtract_x(self.vars.m, -self.vars.mres)

        # smoothed covariance
        self.vars.P += Dmat @ (P_next - Phat) @ Dmat.T
        self.vars.P = symmetrize_mat(self.vars.P, eps=self.epsilon)
        self.vars.Pinv = np.linalg.pinv(self.vars.P, hermitian=True)

        # if data is encountered
        if self.vars.y is not None:
            yhat, Smat, Pxy = self.ut_fun_h.transform(self.vars.m, Phat)
            self.vars.yres = self.fun_subtract_y(self.vars.y, yhat)
            Smat += self.vars.R
            self.vars.Sinv = np.linalg.pinv(Smat, hermitian=True)
            Kmat = Pxy @ self.vars.Sinv
            self.vars.mres = np.dot(Kmat, self.vars.yres)

        # # old
        # mhat, Phat, Cmat = self.ut_fun_f.transform(self.m, self.P)
        # Gmat = self.fun_Gjac(self.m, self.vec_qbar)
        # Qmat = self.fun_Q(self.m, self.vec_qbar)
        # Phat += Gmat @ Qmat @ Gmat.T
        # Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True)
        # xres = Dmat @ self.fun_subtract_x(smean_next, mhat)
        # self.m = self.fun_subtract_x(self.m, -xres)
        # self.P += Dmat @ (scov_next - Phat) @ Dmat.T
        # self.P = self.symmetrize(self.P)
        # self.cur_smetrics = self.compute_metrics()
        # if self.y is not None:
        #     yhat, Smat, Pxy = self.ut_fun_h.transform(self.m, Phat)
        #     Jmat = self.fun_Jjac(self.m, self.vec_rbar)
        #     Smat += Jmat @ self.R @ Jmat.T
        #     yres = self.fun_subtract_y(self.y, yhat)
        #     Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        #     Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        #     Kmat = Pxy @ Smat_inv
        #     xres = np.dot(Kmat, yres)
        #     self.cur_smetrics = self.compute_metrics(
        #         xres, Pmat_inv, yres, Smat_inv)
