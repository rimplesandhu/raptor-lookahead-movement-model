"""Extended Kalman filter class"""
from copy import deepcopy
import numpy as np
from .kalman_filter_base import KalmanFilterBase


class ExtendedKalmanFilter(KalmanFilterBase):
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=invalid-name
    # pylint: disable=not-callable
    """Extended Kalman Filter"""

    def __init__(
        self,
        nx: int,
        ny: int,
        dt: float,
        object_id: str | int = 0,
        dt_tol: float = 0.001
    ):
        super().__init__(nx, ny, dt, object_id, dt_tol)
        self.name = 'EKF'

    def validate(self) -> None:
        """Check if system matrices/functions are initiated"""
        super().validate()
        if self.f is None:
            self.raiseit('Need to define dynamics function f()')
        if self.h is None:
            self.raiseit('Need to define observation function h()')
        if self.compute_F is None:
            self.raiseit('Need to define function compute_F to compute F!')
        if self.compute_G is None:
            self.raiseit('Need to define function compute_G to compute G!')
        if self.compute_Q is None:
            self.raiseit('Need to define function compute_Q to compute Q!')
        if self.compute_H is None:
            self.raiseit('Need to define function compute_H to compute H!')
        if self.compute_J is None:
            self.raiseit('Need to define function compute_J to compute J!')

    def forecast(self) -> None:
        """Kalman filter forecast step"""
        super().forecast()
        F = self.compute_F(self.m, self.qbar)
        #print('F \n', np.array_str(np.array(F), precision=3))
        G = self.compute_G(self.m, self.qbar)
        Q = self.compute_Q()
        self._m = self.f(self.m, self.qbar)
        self._P = F @ self.P @ F.T + G @ Q @ G.T
        self._P = self.symmetrize(deepcopy(self.P))
        #print('forecast \n', np.array_str(np.array(self.P), precision=2))
        #print(f'forecast at {self.time_elapsed}')
        self._store_this_step()

    def update(self) -> None:
        """Kalman filter update step"""
        H = self.compute_H(self.m, self.rbar)
        J = self.compute_J(self.m, self.rbar)
        y_pred = self.h(self.m, self.rbar)
        self._S = H @ self.P @ H.T + J @ self.R @ J.T
        S_inv = np.linalg.pinv(self.S, hermitian=True)
        y_res = self.obs - y_pred
        # if len(y_res) > 2:
        #     y_res[2] = y_res[2] % (2 * np.pi)
        #     if y_res[2] > np.pi:
        #         y_res[2] -= 2 * np.pi
        self._K = self.P @ H.T @ S_inv
        x_res = self.K @ y_res
        self._m += x_res
        # self._m[2] = self._m[2] % (2 * np.pi)
        # if self._m[2] > np.pi:
        #     self._m[2] -= 2 * np.pi
        I_KH = np.eye(self.nx) - np.dot(self.K, H)
        self._P = np.dot(I_KH, self.P).dot(I_KH.T) + \
            np.dot(self.K, self.R).dot(self.K.T)
        self._P = self.symmetrize(self.P) + 0. * np.diag([1e-08] * self.nx)
        #print('update \n', np.array_str(np.array(self.P), precision=2))
        #print(f'update at {self.time_elapsed}')
        # self._P = (np.eye(self.nx) - self.K @ H) @ self.P
        if np.any(np.isnan(self.P)) or np.any(self.P.diagonal() < 0.):
            print('\np update went wrong!')
            print(self.P.diagonal())
        self._nis = np.linalg.multi_dot([y_res.T, S_inv, y_res])
        self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) +
                               np.log(np.linalg.det(self.S)) + self.nis)
        P_inv = np.linalg.pinv(self.P, hermitian=True)
        if self.truth is not None:
            x_res = self.m - self.truth
        self._nees = np.linalg.multi_dot([x_res.T, P_inv, x_res])
        self._store_this_step(update=True)

    def backward_update(self):
        """Backward filter"""
        smean_next = self.history_smoother['state_mean'][-1]
        scov_next = self.history_smoother['state_cov'][-1]
        self.F = self.compute_F(self.m, self.qbar)
        self.G = self.compute_G(self.m, self.qbar)
        self.Q = self.compute_Q()
        fcov = self.F @ self.P @ self.F.T + self.G @ self.Q @ self.G.T
        #print('cond number F before: ', np.linalg.cond(fcov))
        #fcov += np.diag([1e-01] * self.nx)
        #print('cond number F after: ', np.linalg.cond(fcov))
        fcov_inv = np.linalg.pinv(fcov.copy(), hermitian=True)
        # print('inverse check:', fcov @ fcov_inv)
        gmat = self.P @ self.F.T @ fcov_inv
        self._m += gmat @ (smean_next - self.f(self.m, self.qbar))
        self._P += gmat @ (scov_next - fcov) @ gmat.T
        # print('cond number P after: ', np.linalg.cond(self.P))
        # if np.any(np.isnan(self.P)) or np.any(self.P.diagonal() < 0.):
        # print('\np smoother update went wrong!')
        # print(self.P.diagonal())
        # inds = np.where(self.P.diagonal() < 0.)[0]
        # for ind in inds:
        #     self._P[ind, ind] = 1e-5
        if self.obs is not None:
            self.H = self.compute_H(self.m, self.rbar)
            self.J = self.compute_J(self.m, self.rbar)
            yres = self.obs - self.H @ self.m
            self._S = self.H @ self.P @ self.H.T + self.J @ self.R @ self.J.T
            Smat_inv = np.linalg.pinv(self.S, hermitian=True)
            self._nis = np.linalg.multi_dot([yres.T, Smat_inv, yres])
            self._loglik = -0.5 * (self.ny * np.log(2. * np.pi) +
                                   np.log(np.linalg.det(self.S)) + self.nis)
            if self.truth is not None:
                xres = self.m - self.truth
            else:
                Kmat = self.P @ self.H.T @ Smat_inv
                xres = np.dot(Kmat, yres)
            scov_inv = np.linalg.pinv(self.P, hermitian=True)
            self._nees = np.linalg.multi_dot([xres.T, scov_inv, xres])
