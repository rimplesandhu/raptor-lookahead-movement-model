"""Unscented Kalman Filter class"""
import numpy as np
from .kalman_filter_base import KalmanFilterBase
from .unscented_transform import UnscentedTransform


class UnscentedKalmanFilter(KalmanFilterBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    """Unscented Kalman Filter class"""

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0,
        **kwargs
    ):
        super().__init__(nx, ny, dt, object_id)
        self.name = 'UKF'
        self.ut = UnscentedTransform(dim=nx, **kwargs)

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        super().validate()
        if (self.func_f is None) and (self.F is None):
            self.raiseit('UKF: Need to define either function f() or matrix F')
        if (self.func_h is None) and (self.H is None):
            self.raiseit('UKF: Need to define either function h() or matrix H')

    def forecast(self) -> None:
        """UKF forecast step"""
        super().forecast()
        # print('----', self.time_elapsed)
        Qmat = self.func_Q(self.m) if self.Q is None else self.Q
        Qmat = self.symmetrize(Qmat) + np.diag([self.epsilon] * self.nx)
        # print(np.array_str(Qmat, precision=1, suppress_small=True))
        if self.F is None:
            self._m, new_P, _ = self.ut.transform(
                self.m, self.P, self.func_f,
                self.x_subtract, self.x_subtract, self.x_mean_fn
            )
        else:
            self._m = self.F @ self.m + self.qbar
            new_P = self.F @ self.P @ self.F.T
        # print(np.array_str(new_P, precision=1, suppress_small=True))
        new_P += self.G @ Qmat @ self.G.T
        new_P = self.symmetrize(new_P) + np.diag([self.epsilon] * self.nx)
        if not np.any(np.linalg.eigvals(new_P) < 0):
            self._P = new_P
        self._store_this_step()

    def update(self) -> None:
        """UKF update step"""
        if self.H is None:
            y_pred, Smat, Pxy = self.ut.transform(
                self.m, self.P, self.func_h,
                self.x_subtract, self.y_subtract, self.y_mean_fn
            )
        else:
            y_pred = self.H @ self.m
            Smat = self.H @ self.P @ self.H.T
            Pxy = self.P @ self.H.T
        Smat += self.J @ self.R @ self.J.T
        y_res = self.y_subtract(self.obs, y_pred)
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        Kmat = Pxy @ Smat_inv
        x_res = Kmat @ y_res
        self._m = self.x_add(self.m, x_res)
        Tmat = np.eye(self.nx) - Kmat @ self.H
        self._P = Tmat @ self.P @ Tmat.T + Kmat @ self.R @ Kmat.T  # Joseph
        # self._P -= Kmat @ Smat @ Kmat.T
        self._P = self.symmetrize(self.P) + np.diag([self.epsilon] * self.nx)
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True, rcond=1e-10)
        self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
        self._store_this_step(update=True)

    def _backward_filter(self, smean_next, scov_next):
        """Backward filter"""
        if self.F is None:
            mhat, Phat, Cmat = self.ut.transform(
                self.m, self.P, self.func_f,
                self.x_subtract, self.x_subtract, self.x_mean_fn)
        else:
            mhat = self.F @ self.m
            Phat = self.F @ self.P @ self.F.T
            Cmat = self.P @ self.F.T
        Qmat = self.func_Q(self.m) if self.Q is None else self.Q
        Phat += self.G @ Qmat @ self.G.T
        Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True, rcond=1e-10)
        Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True)
        self._m = self.x_add(self.m, Dmat @ self.x_subtract(smean_next, mhat))
        self._P += Dmat @ (scov_next - Phat) @ Dmat.T
        self._P = self.symmetrize(self.P) + np.diag([self.epsilon] * self.nx)
        if self.obs is not None:
            if self.H is None:
                y_pred, Smat, Pxy = self.ut.transform(
                    self.m, Phat, self.func_h,
                    self.x_subtract, self.y_subtract, self.y_mean_fn)
            else:
                y_pred = self.H @ self.m
                Smat = self.H @ Phat @ self.H.T
                Pxy = Phat @ self.H.T
            Smat += self.J @ self.R @ self.J.T
            y_res = self.y_subtract(self.obs, y_pred)
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            Kmat = Pxy @ Smat_inv
            x_res = np.dot(Kmat, y_res)
            self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
