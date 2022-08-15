"""Unscented Kalman Filter class"""
from collections.abc import Callable
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
        if self.f is None:
            self.raiseit('Need to define dynamics function f()')
        if self.h is None:
            self.raiseit('Need to define observation function h()')
        if self.Q is None:
            self.raiseit('Need to initiate Q matrix!')

    def forecast(self) -> None:
        """UKF forecast step"""
        super().forecast()
        #print('----', self.time_elapsed)
        self._m, self._P, _ = self.ut.transform(self.m, self.P, self.f,
                                                self.x_subtract,
                                                self.x_subtract,
                                                self.x_mean_fn)

        # if np.any(np.linalg.eigvals(self.P) < 0):
        #     print('P gone wrong forecast:',
        #           self.time_elapsed, self.last_update_at)
        self._P += self.G @ self.Q @ self.G.T
        # if np.any(np.linalg.eigvals(self.P) < 0):
        #     print('P gone forecast wrong!')
        #print('Forecast:', np.degrees(self.m[2]), np.degrees(self.P[2, 2]))
        self._store_this_step()

    def update(self) -> None:
        """UKF update step"""
        y_pred, Smat, Pxy = self.ut.transform(self.m, self.P, self.h,
                                              self.x_subtract,
                                              self.y_subtract,
                                              self.y_mean_fn)
        #print('y_pred:', np.around(y_pred, 2))
        y_res = self.y_subtract(self.obs, y_pred)
        #print('y_res:', np.around(y_res, 2))
        #print('yobs,ypred', np.degrees(self.obs[2]), np.degrees(y_pred[2]))
        Smat += self.R
        Smat_inv = np.linalg.pinv(Smat, hermitian=True)
        Kmat = Pxy @ Smat_inv
        x_res = Kmat @ y_res
        #print('x_res:', np.around(x_res, 2))
        self._m = self.x_add(self.m, x_res)
        #print('obs:', np.around(self.obs, 2))
        #print('obs r:', np.around(self.R.diagonal(), 2))
        #print('update m:', np.around(self.m, 2))
        #print('update:', np.degrees(x_res[2]), np.degrees(self.m[2]))
        # self._m = self.x_subtract(self.m, -x_res)
        self._P -= Kmat @ Smat @ Kmat.T
        self._P = self.symmetrize(self.P) + 0. * \
            np.diag([self.epsilon] * self.nx)
        #print('update P:', np.around(self.P.diagonal(), 2))
        Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
        self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
        self._store_this_step(update=True)

    def _backward_filter(self):
        """Backward filter"""
        smean_next = self.history['smoother_mean'][-1]
        scov_next = self.history['smoother_cov'][-1]
        mhat, Phat, Cmat = self.ut.transform(
            self.m, self.P, self.f,
            self.x_subtract, self.x_subtract, self.x_mean_fn)
        Phat += self.G @ self.Q @ self.G.T
        Dmat = Cmat @ np.linalg.pinv(Phat, hermitian=True)
        #self._m += Dmat @ self.x_subtract(smean_next, mhat)
        self._m = self.x_add(self.m, Dmat @ self.x_subtract(smean_next, mhat))
        # self._m += Dmat @ (smean_next - mhat)
        self._P += Dmat @ (scov_next - Phat) @ Dmat.T
        self._P = self.symmetrize(self.P) + 0. * \
            np.diag([self.epsilon] * self.nx)
        if self.obs is not None:
            y_pred, Smat, Pxy = self.ut.transform(self.m, self.P, self.h,
                                                  self.x_subtract,
                                                  self.y_subtract,
                                                  self.y_mean_fn)
            y_res = self.y_subtract(self.obs, y_pred)
            Smat += self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(Smat, hermitian=True)
            Pmat_inv = np.linalg.pinv(self.P, hermitian=True)
            Kmat = Pxy @ Smat_inv
            x_res = np.dot(Kmat, y_res)
            self._compute_metrics(x_res, Pmat_inv, y_res, Smat_inv)
