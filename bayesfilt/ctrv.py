"""Classes defining Constant Turn Rate and Velocity Models"""
# pylint: disable=invalid-name
from copy import deepcopy
from functools import partial
import numpy as np
from numpy import ndarray
from .motion_model import MotionModel
from .utils import subtract_states, state_mean_func, symmetrize


class CTRV_POINT(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=5, name='CTRV_POINT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed', 'HeadingRate']
        self.subtract_states = partial(subtract_states, angle_idx=2)
        self.state_mean_func = partial(state_mean_func, angle_idx=2)

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_vel']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        # x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        next_x[2] = x[2] + x[4] * self.dt
        next_x[3] = x[3]
        next_x[4] = x[4]
        # if x[3] < 2.:  # low speed
        #     next_x[3] = 0.
        #     next_x[2] = x[2]
        if np.abs(x[4]) > 0.001:
            next_x[0] = x[0] + x[3] / x[4] * (np.sin(next_x[2]) - np.sin(x[2]))
            next_x[1] = x[1] - x[3] / x[4] * (np.cos(next_x[2]) - np.cos(x[2]))
        else:
            next_x[0] = x[0] + x[3] * np.cos(x[2]) * self.dt
            next_x[1] = x[1] + x[3] * np.sin(x[2]) * self.dt
        return next_x

    def func_Q(
        self,
        x: ndarray
    ) -> ndarray:
        """Get Q matrix"""
        ssq = [self.phi['sigma_vel'], self.phi['sigma_omega']]
        ssq = [ix**2 for ix in ssq]
        self._Q[0, 0] = ssq[0] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
        self._Q[0, 1] = 1. * ssq[0] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
        self._Q[0, 3] = 1. * ssq[0] * np.cos(x[2]) * self.dt ** 2 / 2.
        self._Q[0, 4] = 1. * ssq[1] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
        self._Q[1, 1] = ssq[0] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
        self._Q[1, 3] = 1 * ssq[0] * np.sin(x[2]) * self.dt ** 2 / 2.
        self._Q[1, 4] = 1 * ssq[1] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
        self._Q[2, 2] = ssq[1] * self.dt ** 3 / 3.
        self._Q[2, 4] = ssq[1] * self.dt ** 2 / 2.
        self._Q[3, 3] = ssq[0] * self.dt ** 1 / 1.
        self._Q[4, 4] = ssq[1] * self.dt ** 1 / 1.
        self._Q = symmetrize(self.Q)
        return self.Q


class CTRV_RECT(MotionModel):
    """Class for Constant Turn Rate and Velocity for a 2D object"""

    def __init__(self):
        super().__init__(nx=7, name='CTRV_RECT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed', 'HeadingRate',
                            'Width', 'Length']
        self._ctrv = CTRV_POINT()
        self.subtract_states = partial(subtract_states, angle_idx=2)
        self.state_mean_func = partial(state_mean_func, angle_idx=2)

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_vel', 'sigma_w', 'sigma_l']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        # x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        next_x[0:5] = self._ctrv.func_f(x[0:5])
        next_x[5] = x[5]
        next_x[6] = x[6]
        return next_x

    def func_Q(
        self,
        x: ndarray
    ) -> None:
        """Get Q matrix"""
        ssq = [self.phi['sigma_vel'], self.phi['sigma_omega'],
               self.phi['sigma_w'], self.phi['sigma_l']]
        self._ctrv.phi = {k: v for k,
                          v in self.phi.items() if k in self._ctrv.phi_names}
        self._ctrv.dt = self.dt
        ssq = [ix**2 for ix in ssq]
        self._Q = np.eye(self.nx)
        self._Q[0:5, 0:5] = self._ctrv.func_Q(x[0:5])
        self._Q[5, 5] = ssq[2] * self.dt ** 2 / 1.
        self._Q[6, 6] = ssq[3] * self.dt ** 2 / 1.
        return self.Q


class CTRA_POINT(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=6, name='CTRA_POINT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed',
                            'HeadingRate', 'Acceleration']
        self.subtract_states = partial(subtract_states, angle_idx=2)
        self.state_mean_func = partial(state_mean_func, angle_idx=2)

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_accn']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        # x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        if x[3] < 1.75:
            next_x[2] = x[2]
            next_x[3] = x[3]
            next_x[4] = 0.
            next_x[5] = 0.
            next_x[0] = x[0]
            next_x[1] = x[1]
        else:
            next_x[2] = x[2] + x[4] * self.dt
            next_x[3] = x[3] + x[5] * self.dt
            next_x[4] = x[4]
            next_x[5] = x[5]
        # if x[3] < 2.:  # low speed
        #     next_x[3] = 0.
        #     next_x[2] = x[2]
            if np.abs(x[4]) > 0.001:
                next_x[0] = x[0] + x[3] / x[4] * \
                    (np.sin(next_x[2]) - np.sin(x[2]))
                next_x[1] = x[1] - x[3] / x[4] * \
                    (np.cos(next_x[2]) - np.cos(x[2]))
            else:
                next_x[0] = x[0] + x[3] * np.cos(x[2]) * self.dt
                next_x[1] = x[1] + x[3] * np.sin(x[2]) * self.dt
        return next_x

    def func_Q(
        self,
        x: ndarray
    ) -> ndarray:
        """Get Q matrix"""
        ssq = [self.phi['sigma_accn'], self.phi['sigma_omega']]
        ssq = [ix**2 for ix in ssq]
        self._Q = np.zeros((self.nx, self.nx))
        # self._Q[0, 0] = ssq[0] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
        # self._Q[0, 1] = 1. * ssq[0] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
        # self._Q[0, 3] = 1. * ssq[0] * np.cos(x[2]) * self.dt ** 2 / 2.
        # self._Q[0, 4] = 1. * ssq[1] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
        # self._Q[1, 1] = ssq[0] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
        # self._Q[1, 3] = 1 * ssq[0] * np.sin(x[2]) * self.dt ** 2 / 2.
        # self._Q[1, 4] = 1 * ssq[1] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
        self._Q[2, 2] = ssq[1] * self.dt ** 3 / 3.
        self._Q[2, 4] = ssq[1] * self.dt ** 2 / 2.
        self._Q[3, 3] = ssq[0] * self.dt ** 3 / 3.
        self._Q[3, 5] = ssq[0] * self.dt ** 2 / 2.
        self._Q[4, 4] = ssq[1] * self.dt ** 1 / 1.
        self._Q[5, 5] = ssq[0] * self.dt ** 1 / 1.
        self._Q = symmetrize(self.Q)
        return self.Q


class CTRA_RECT(MotionModel):
    """Class for Constant Turn Rate and Accn for a 2D object"""

    def __init__(self):
        super().__init__(nx=8, name='CTRA_RECT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed', 'HeadingRate',
                            'Acceleration', 'Width', 'Length']
        self._ctra = CTRA_POINT()
        self.subtract_states = partial(subtract_states, angle_idx=2)
        self.state_mean_func = partial(state_mean_func, angle_idx=2)

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_accn', 'sigma_w', 'sigma_l']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        # x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        next_x[0:6] = self._ctra.func_f(x[0:6])
        next_x[6] = x[6]
        next_x[7] = x[7]
        return next_x

    def func_Q(
        self,
        x: ndarray
    ) -> None:
        """Get Q matrix"""
        ssq = [self.phi['sigma_accn'], self.phi['sigma_omega'],
               self.phi['sigma_w'], self.phi['sigma_l']]
        self._ctra.phi = {k: v for k,
                          v in self.phi.items() if k in self._ctra.phi_names}
        self._ctra.dt = self.dt
        ssq = [ix**2 for ix in ssq]
        self._Q = np.eye(self.nx)
        self._Q[0:6, 0:6] = self._ctra.func_Q(x[0:6])
        self._Q[7, 7] = ssq[3] * self.dt ** 1 / 1.
        self._Q[6, 6] = ssq[2] * self.dt ** 1 / 1.
        return self.Q


# class CTRV3D(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a 3D object"""

#     def __init__(self):
#         super().__init__(nx=8, nq=8, name='CTRV3D')
#         self._ctrv = CTRV()

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         x = self.vec_setter(x, self.nx)
#         next_x = x.copy()
#         next_x[0:5] = self._ctrv.f(x[0:5])
#         next_x[5] = x[5]
#         next_x[6] = x[6]
#         next_x[7] = x[7]
#         if q is not None:
#             q = self.vec_setter(q, self.nx)
#             next_x += q
#         if u is not None:
#             u = self.vec_setter(u, self.nx)
#             next_x += u
#         return next_x

#     def update(
#         self,
#         dt: float,
#         sigmas: ndarray,
#         w: ndarray | None = None
#     ) -> None:
#         """Update system parameters and matrices"""
#         self._dt = self.float_setter(dt)
#         self._sigmas = self.vec_setter(sigmas, 5)
#         self._ctrv.update(dt, sigmas[0:2])

#     def compute_F(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         out_F = np.eye(self.nx)
#         out_F[0:5, 0:5] = self._ctrv.compute_F(x[0:5])
#         return self.F

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         ssq = self.sigmas**2
#         self.Q = np.eye(self.nx)
#         self.Q[0:5, 0:5] = self._ctrv.compute_Q(x[0:5])
#         self.Q[5, 5] = ssq[1] * self.dt ** 2 / 1.
#         self.Q[6, 6] = ssq[2] * self.dt ** 2 / 1.
#         self.Q[7, 7] = ssq[3] * self.dt ** 2 / 1.
#         return self.Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)

    # def func_Q(
    #     self,
    #     x: ndarray | None,
    #     q: ndarray | None = None
    # ) -> None:
    #     """Get F matrix"""

    #     # common terms
    #     atmp = x[3] / x[4] * (np.sin(x[2] + x[4] * self.dt) - np.sin(x[2]))
    #     btmp = x[3] / x[4] * (np.cos(x[2] + x[4] * self.dt) - np.cos(x[2]))
    #     ctmp = (x[3] * self.dt / x[4]) * np.sin(x[4] * self.dt + x[2])
    #     dtmp = (x[3] * self.dt / x[4]) * np.cos(x[4] * self.dt + x[2])

    #     # Jacobian matrix F
    #     out_F = np.eye(self.nx)
    #     out_F[0, 2] = btmp
    #     out_F[0, 3] = atmp / x[3]
    #     out_F[0, 4] = -atmp / x[4] + dtmp
    #     out_F[1, 2] = atmp
    #     out_F[1, 3] = -btmp / x[3]
    #     out_F[1, 4] = btmp / x[4] + ctmp
    #     out_F[2, 4] = self.dt
    #     return out_F
