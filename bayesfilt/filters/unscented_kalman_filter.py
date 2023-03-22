"""Unscented Kalman Filter class"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
import numpy as np
from .kalman_filter_base import KalmanFilterBase
from .unscented_transform import UnscentedTransform


class UnscentedKalmanFilter(KalmanFilterBase):
    """Unscented Kalman Filter"""

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert 'alpha' in self.pars, self.raiseit('Need alpha in pars of UKF!')
        assert 'beta' in self.pars, self.raiseit('Need beta in pars of UKF!')
        kappa = self.pars['kappa'] if 'kappa' in self.pars else None
        use_chol = self.pars['use_cholesky'] if 'use_cholesky' in self.pars else None
        self.ut_fun_f = UnscentedTransform(
            dim=self.nx,
            alpha=self.pars['alpha'],
            beta=self.pars['beta'],
            kappa=kappa,
            use_cholesky=use_chol,
            fun_subtract_in=self.fun_subtract_x,
            model_fun=self.fun_f,
            fun_subtract_out=self.fun_subtract_x,
            fun_mean_out=self.fun_weighted_mean_x
        )
        self.ut_fun_h = UnscentedTransform(
            dim=self.nx,
            alpha=self.pars['alpha'],
            beta=self.pars['beta'],
            kappa=kappa,
            use_cholesky=use_chol,
            fun_subtract_in=self.fun_subtract_x,
            model_fun=self.fun_h,
            fun_subtract_out=self.fun_subtract_y,
            fun_mean_out=self.fun_weighted_mean_y
        )

    def initiate(self, *args, **kwargs):
        """Initiate function"""
        super().initiate(*args, **kwargs)
        if self.fun_f is None:
            self.raiseit('Need dynamics function fun_f to initiate UKF!')
        if self.fun_h is None:
            self.raiseit('Need obs function fun_h to initiate UKF!')
        if self.fun_Q is None:
            self.raiseit('Need function fun_Q to initiate UKF!')

    def forecast(self) -> None:
        """UKF forecast step"""
        super().forecast()
        self.m, self.P, _ = self.ut_fun_f.transform(self.m, self.P)
        Qmat = self.fun_Q(self.m, self.vec_qbar)
        Gmat = self.fun_Gjac(self.m, self.vec_qbar)
        self.P += Gmat @ Qmat @ Gmat.T
        self.P = self.symmetrize(self.P)
        self.store_this_timestep()

    def update(self) -> None:
        """UKF update step"""

        yhat, Smat, Pxy = self.ut_fun_h.transform(self.m, self.P)
        Hmat = self.fun_Hjac(self.m, self.vec_rbar)
        Jmat = self.fun_Jjac(self.m, self.vec_rbar)
        Smat += Jmat @ self.R @ Jmat.T
        yres = self.fun_subtract_y(self.y, yhat)
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        Kmat = Pxy @ Smat_inv
        xres = Kmat @ yres
        self.m = self.fun_subtract_x(self.m, -xres)
        Tmat = np.eye(self.nx) - Kmat @ Hmat
        self.P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T  # Joseph
        # self._P -= Kmat @ Smat @ Kmat.T
        self.P = self.symmetrize(self.P)
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        self.cur_metrics = self.compute_metrics(
            xres, Pmat_inv, yres, Smat_inv)
        self.store_this_timestep(update=True)

    def _backward_filter(self, smean_next, scov_next):
        """Backward filter"""
        mhat, Phat, Cmat = self.ut_fun_f.transform(self.m, self.P)
        Gmat = self.fun_Gjac(self.m, self.vec_qbar)
        Qmat = self.fun_Q(self.m, self.vec_qbar)
        Phat += Gmat @ Qmat @ Gmat.T
        Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True)
        xres = Dmat @ self.fun_subtract_x(smean_next, mhat)
        self.m = self.fun_subtract_x(self.m, -xres)
        self.P += Dmat @ (scov_next - Phat) @ Dmat.T
        self.P = self.symmetrize(self.P)
        self.cur_smetrics = self.compute_metrics()
        if self.y is not None:
            yhat, Smat, Pxy = self.ut_fun_h.transform(self.m, Phat)
            Jmat = self.fun_Jjac(self.m, self.vec_rbar)
            Smat += Jmat @ self.R @ Jmat.T
            yres = self.fun_subtract_y(self.y, yhat)
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            Kmat = Pxy @ Smat_inv
            xres = np.dot(Kmat, yres)
            self.cur_smetrics = self.compute_metrics(
                xres, Pmat_inv, yres, Smat_inv)
