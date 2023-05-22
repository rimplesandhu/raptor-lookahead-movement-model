"""Classes defining Constant Turn Rate and Velocity/Acceleration Models"""
# pylint: disable=invalid-name
from copy import deepcopy
from functools import partial
import numpy as np
from numpy import ndarray
from .motion_model import MotionModel


class CTRV_POINT(MotionModel):
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=5, name='CTRV_POINT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed', 'HeadingRate']

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
        self._Q = self.symmetrize(self.Q)
        return self.Q


class CTRV_RECT(MotionModel):
    """Class for Constant Turn Rate and Velocity for a 2D object"""

    def __init__(self):
        super().__init__(nx=7, name='CTRV_RECT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed', 'HeadingRate',
                            'Width', 'Length']
        self._ctrv = CTRV_POINT()

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_vel', 'sigma_w', 'sigma_l']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
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

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_accn', 'min_speed']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        next_x = np.asarray(deepcopy(x))
        if x[3] < self.phi['min_speed']:
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
        self._Q = self.symmetrize(self.Q)
        return self.Q


class CTRA_RECT(MotionModel):
    """Class for Constant Turn Rate and Accn for a 2D object"""

    def __init__(self):
        super().__init__(nx=8, name='CTRA_RECT')
        self.state_names = ['X', 'Y', 'Heading', 'Speed', 'HeadingRate',
                            'Acceleration', 'Width', 'Length']
        self._ctra = CTRA_POINT()

    @property
    def phi_names(self):
        return ['sigma_omega', 'sigma_accn', 'min_speed'] + ['sigma_w', 'sigma_l']

    def func_f(
        self,
        x: ndarray,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
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
#         next_x = x.copy()
#         next_x[0:5] = self._ctrv.f(x[0:5])
#         next_x[5] = x[5]
#         next_x[6] = x[6]
#         next_x[7] = x[7]
#         if q is not None:
#             next_x += q
#         if u is not None:
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


# class CTRA2D(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity in 2D"""

#     def __init__(self, fac=0.95):
#         super().__init__(nx=6, nq=6, name='CTRA2D')
#         self.labels = ['X', 'Y', 'Heading', 'Hspeed', 'HeadingRate', 'Haccn']
#         self.factor = fac

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         next_x = deepcopy(x)
#         # next_x[4] = min(abs(next_x[4]), np.pi / 7.) * \
#         #     (next_x[4] / abs(next_x[4]))

#         # if x[3] < 0.:
#         #     x[3] *= -1
#         #     x[2] += np.pi

#         next_x[4] = self.factor * x[4]  # yawrate
#         next_x[5] = self.factor * x[5]  # hor accn

#         #next_x[3] = abs(next_x[3])
#         # next_x[0] += x[3] * np.sin(x[2]) * self.dt  # x loc
#         # next_x[1] += x[3] * np.cos(x[2]) * self.dt  # y loc

#         # else:
#         # if abs(x[4]) > np.radians(5.):
#         next_x[2] += x[4] * self.dt  # heading
#         next_x[3] += x[5] * self.dt  # speed

#         #next_x[3] = abs(next_x[3])
#         next_x[2] = next_x[2] % (2.0 * np.pi)
#         if next_x[2] > np.pi:
#             next_x[2] -= 2. * np.pi
#         t1 = 1 / (x[4]**2)
#         next_x[1] += t1 * (next_x[3] * x[4] * np.sin(next_x[2])
#                            - x[3] * x[4] * np.sin(x[2])
#                            + x[5] * np.cos(next_x[2])
#                            - x[5] * np.cos(x[2]))

#         next_x[0] += t1 * (-next_x[3] * x[4] * np.cos(next_x[2])
#                            + x[3] * x[4] * np.cos(x[2])
#                            + x[5] * np.sin(next_x[2])
#                            - x[5] * np.sin(x[2]))

#         #curv = next_x[4] / next_x[3]

#         if q is not None:
#             next_x += q
#         if u is not None:
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

#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""

#         # common terms
#         # atmp = x[3] / x[4] * (np.sin(x[2] + x[4] * self.dt) - np.sin(x[2]))
#         # btmp = x[3] / x[4] * (np.cos(x[2] + x[4] * self.dt) - np.cos(x[2]))
#         # ctmp = (x[3] * self.dt / x[4]) * np.sin(x[4] * self.dt + x[2])
#         # dtmp = (x[3] * self.dt / x[4]) * np.cos(x[4] * self.dt + x[2])
#         # Jacobian matrix F
#         out_F = np.eye(self.nx)
#         out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
#         out_F[0, 3] = np.cos(x[2]) * self.dt
#         out_F[1, 2] = x[3] * np.cos(x[2]) * self.dt
#         out_F[1, 3] = np.sin(x[2]) * self.dt
#         out_F[2, 4] = self.dt
#         out_F[3, 5] = self.dt
#         return out_F

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.zeros((self.nx, self.nx))
#         # if x is not None:
#         #     out_Q[0, 0] = sigma_sq[1] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
#         #     out_Q[0, 1] = sigma_sq[1] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
#         #     out_Q[0, 3] = sigma_sq[1] * np.cos(x[2]) * self.dt ** 2 / 2.
#         #     out_Q[0, 4] = sigma_sq[0] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
#         #     out_Q[1, 1] = sigma_sq[1] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
#         #     out_Q[1, 3] = sigma_sq[1] * np.sin(x[2]) * self.dt ** 2 / 2.
#         #     out_Q[1, 4] = sigma_sq[0] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
#         # out_Q[0, 0] = -sigma_sq[0] * self.dt ** 3 / 3.
#         # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         # # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
#         # # out_Q[3, 5] = sxsigma_sq[1] * self.dt ** 2 / 2.
#         #
#         out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.

#         out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
#         out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
#         out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.

#         return out_Q


#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)


# class CTRA3D(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a point object"""

#     def __init__(self):
#         super().__init__(nx=9, nq=9, name='CTRA')
#         self.labels = ['X', 'Y', 'Heading', 'Hspeed', 'HeadingRate', 'Haccn',
#                        'Altitude', 'Vspeed', 'Vaccn']

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         next_x = deepcopy(x)
#         next_x[2] += x[4] * self.dt
#         next_x[3] += x[5] * self.dt
#         next_x[0] += x[3] * np.sin(x[2]) * self.dt
#         next_x[1] += x[3] * np.cos(x[2]) * self.dt
#         #     t1 = 1 / (x[4]**2)
#         #     next_x[1] += t1 * (next_x[3] * x[4] * np.sin(next_x[2])
#         #                     - x[3] * x[4] * np.sin(x[2])
#         #                     + x[5] * np.cos(next_x[2])
#         #                     - x[5] * np.cos(x[2]))

#         #     next_x[0] += t1 * (-next_x[3] * x[4] * np.cos(next_x[2])
#         #                     + x[3] * x[4] * np.cos(x[2])
#         #                     + x[5] * np.sin(next_x[2])
#         #                     - x[5] * np.sin(x[2]))

#         next_x[6] += x[7] * self.dt + 0.5 * x[8] * self.dt**2
#         next_x[7] += x[8] * self.dt

#         if q is not None:
#             next_x += q
#         if u is not None:
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
#         self._sigmas = self.vec_setter(sigmas, 3)

#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         # atmp = x[3] / x[4] * (np.sin(x[2] + x[4] * self.dt) - np.sin(x[2]))
#         # btmp = x[3] / x[4] * (np.cos(x[2] + x[4] * self.dt) - np.cos(x[2]))
#         # ctmp = (x[3] * self.dt / x[4]) * np.sin(x[4] * self.dt + x[2])
#         # dtmp = (x[3] * self.dt / x[4]) * np.cos(x[4] * self.dt + x[2])
#         out_F = np.eye(self.nx)
#         out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
#         out_F[0, 3] = np.cos(x[2]) * self.dt
#         out_F[1, 2] = x[3] * np.cos(x[2]) * self.dt
#         out_F[1, 3] = np.sin(x[2]) * self.dt
#         out_F[2, 4] = self.dt
#         out_F[3, 5] = self.dt
#         return out_F

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.zeros((self.nx, self.nx))
#         # if x is not None:
#         #     out_Q[0, 0] = sigma_sq[1] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
#         #     out_Q[0, 1] = sigma_sq[1] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
#         #     out_Q[0, 3] = sigma_sq[1] * np.cos(x[2]) * self.dt ** 2 / 2.
#         #     out_Q[0, 4] = sigma_sq[0] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
#         #     out_Q[1, 1] = sigma_sq[1] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
#         #     out_Q[1, 3] = sigma_sq[1] * np.sin(x[2]) * self.dt ** 2 / 2.
#         #     out_Q[1, 4] = sigma_sq[0] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
#         # out_Q[0, 0] = -sigma_sq[0] * self.dt ** 3 / 3.
#         # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         # # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
#         # # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
#         #

#         out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.

#         out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
#         out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
#         out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.

#         out_Q[6, 6] = sigma_sq[2] * self.dt**5 / 20
#         out_Q[6, 7] = sigma_sq[2] * self.dt**4 / 8
#         out_Q[7, 6] = sigma_sq[2] * self.dt**4 / 8
#         out_Q[6, 8] = sigma_sq[2] * self.dt**3 / 6
#         out_Q[8, 6] = sigma_sq[2] * self.dt**3 / 6
#         out_Q[7, 7] = sigma_sq[2] * self.dt**3 / 3
#         out_Q[7, 8] = sigma_sq[2] * self.dt**2 / 2
#         out_Q[8, 7] = sigma_sq[2] * self.dt**2 / 2
#         out_Q[8, 8] = sigma_sq[2] * self.dt**1 / 1
#         return out_Q

#     def subtract(self, x0: ndarray, x1: ndarray):
#         """Residual function for computing difference among states"""
#         x0 = self.vec_setter(x0, self.nx)
#         x1 = self.vec_setter(x1, self.nx)
#         xres = x0 - x1
#         xres[2] = xres[2] % (2.0 * np.pi)
#         if xres[2] > np.pi:
#             xres[2] -= 2. * np.pi
#         return xres

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)


# class CCA(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a point object"""

#     def __init__(self):
#         super().__init__(nx=6, nq=6, name='CCA')
#         self._labels = ['Position X', 'Position Y', 'Heading', 'Speed',
#                         'Acceleration', 'Rcurvature']

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         x = self.vec_setter(x, self.nx)
#         next_x = deepcopy(x)
#         next_x[3] += x[4] * self.dt  # speed
#         if abs(x[5]) < 20000.:
#             next_x[2] += (next_x[3]**2 - x[3]**2) / (2. * x[4] * x[5])
#             next_x[1] += x[5] * (np.sin(next_x[2]) - np.sin(x[2]))  # x
#             next_x[0] += -x[5] * (np.cos(next_x[2]) - np.cos(x[2]))  # y
#         else:
#             next_x[1] += (next_x[3]**2 - x[3]**2) * np.cos(x[2]) / (2. * x[4])
#             next_x[0] += (next_x[3]**2 - x[3]**2) * np.sin(x[2]) / (2. * x[4])
#         # if next_x[2] > np.pi:
#         #     next_x[2] -= 2.0 * np.pi
#         # next_x[3] += x[4] * self.dt  # speed

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
#         self._sigmas = self.vec_setter(sigmas, 2)

#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         out_F = np.eye(self.nx)
#         return out_F

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.zeros((self.nx, self.nx))
#         out_Q[2, 2] = (10 * np.pi / 180)**2
#         out_Q[3, 3] = sigma_sq[0] * self.dt ** 3 / 3.
#         out_Q[3, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 3] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
#         return out_Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)

#     def subtract(self, x0: ndarray, x1: ndarray):
#         """Residual function for computing difference in angle"""
#         x0 = self.vec_setter(x0, self.nx)
#         x1 = self.vec_setter(x1, self.nx)
#         xres = x0 - x1
#         xres[2] = xres[2] % (2.0 * np.pi)
#         if xres[2] > np.pi:
#             xres[2] -= 2. * np.pi
#         return xres

#     @ property
#     def labels(self):
#         """Return labels for plotting"""
#         return ['Position X', 'Position Y', 'Heading', 'Speed',
#                 'Acceleration', 'Rcurv']


# class CCAold(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a point object"""

#     def __init__(self):
#         super().__init__(nx=6, nq=6, name='CTRA')
#         self._labels = ['Position X', 'Position Y', 'Heading', 'Speed',
#                         'Acceleration', 'Curvature']

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         x = self.vec_setter(x, self.nx)
#         next_x = deepcopy(x)

#         next_x[2] += -x[5] * (x[3] * self.dt +
#                               0.5 * x[4] * self.dt**2)  # yaw
#         next_x[2] = next_x[2] % (2.0 * np.pi)
#         if next_x[2] > np.pi:
#             next_x[2] -= 2.0 * np.pi
#         next_x[3] += x[4] * self.dt  # speed

#         next_x[0] += next_x[3] * np.cos(x[2] - x[5] * next_x[3] * self.dt)  # x
#         next_x[1] += next_x[3] * np.sin(x[2] - x[5] * next_x[3] * self.dt)  # y

#         # next_x[0] += abs(next_x[3]) * np.cos(next_x[2])  # x
#         # ext_x[1] += abs(next_x[3]) * np.sin(next_x[2])  # y

#         # next_x[0] += abs(x[3]) * np.cos(x[2])  # x
#         # next_x[1] += abs(x[3]) * np.sin(x[2])  # y

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
#         self._sigmas = self.vec_setter(sigmas, 2)

#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         out_F = np.eye(self.nx)
#         return out_F

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.zeros((self.nx, self.nx))
#         # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
#         # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
#         # out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.
#         # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
#         out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
#         return out_Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)

#     def subtract(self, x0: ndarray, x1: ndarray):
#         """Residual function for computing difference in angle"""
#         x0 = self.vec_setter(x0, self.nx)
#         x1 = self.vec_setter(x1, self.nx)
#         xres = x0 - x1
#         xres[2] = xres[2] % (2.0 * np.pi)
#         if xres[2] > np.pi:
#             xres[2] -= 2. * np.pi
#         return xres

#     @ property
#     def labels(self):
#         """Return labels for plotting"""
#         return ['Position X', 'Position Y', 'Heading', 'Speed',
#                 'Angular Speed', 'Acceleration']


# class CTRV2D(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a 2D object"""

#     def __init__(self):
#         super().__init__(nx=7, nq=7, name='CTRV2D')
#         self._ctrv = CTRA()

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
#         self._sigmas = self.vec_setter(sigmas, 4)
#         self._ctrv.update(dt, sigmas[0:2])

#     def compute_F(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         out_F = np.eye(self.nx)
#         out_F[0:5, 0:5] = self._ctrv.compute_F(x[0:5])
#         return out_F

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.eye(self.nx)
#         out_Q[0:5, 0:5] = self._ctrv.compute_Q(x[0:5])
#         out_Q[5, 5] = sigma_sq[2] * self.dt ** 2 / 1.
#         out_Q[6, 6] = sigma_sq[3] * self.dt ** 2 / 1.
#         return out_Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)


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
#         sigma_sq = self.sigmas**2
#         out_Q = np.eye(self.nx)
#         out_Q[0:5, 0:5] = self._ctrv.compute_Q(x[0:5])
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 2 / 1.
#         out_Q[6, 6] = sigma_sq[2] * self.dt ** 2 / 1.
#         out_Q[7, 7] = sigma_sq[3] * self.dt ** 2 / 1.
#         return out_Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)


# class CTRA2(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a point object"""

#     def __init__(self):
#         super().__init__(nx=6, nq=6, name='CTRA2')
#         self._labels = ['Position X', 'Position Y', 'SpeedX', 'SpeedY',
#                         'Angular speed', 'Acceleration']

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         x = self.vec_setter(x, self.nx)
#         next_x = deepcopy(x)

#         v = np.sqrt(x[2]**2 + x[3]**2)
#         next_x[0] += x[2] * self.dt + (0.5 * x[5] * x[2] / v) * self.dt**2
#         next_x[1] += x[3] * self.dt + (0.5 * x[5] * x[3] / v) * self.dt**2
#         t1 = x[5] / (v * x[4])
#         next_x[2] += t1 * (x[3] * (np.cos(x[4] * self.dt) - 1)
#                            + x[2] * np.sin(x[4] * self.dt))
#         next_x[3] += -t1 * (x[2] * (np.cos(x[4] * self.dt) - 1)
#                             - x[3] * np.sin(x[4] * self.dt))

#         #
#         # next_x[2] += (x[5] * x[2] / v - x[4] * x[3]) * self.dt
#         # next_x[3] += (x[5] * x[3] / v + x[4] * x[2]) * self.dt
#         # next_x[2] += x[4] * self.dt
#         # next_x[3] += x[5] * self.dt
#         # if next_x[4] > np.pi:
#         #     next_x[4] -= 2 * np.pi
#         # if next_x[4] < -np.pi:
#         #     next_x[4] += 2 * np.pi
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
#         self._sigmas = self.vec_setter(sigmas, 2)

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.zeros((self.nx, self.nx))

#         # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.

#         # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
#         # out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.

#         # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.

#         out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
#         return out_Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)

#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         out_F = np.eye(self.nx)
#         # out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
#         # out_F[0, 3] = np.cos(x[2]) * self.dt
#         # out_F[1, 2] = x[3] * np.sin(x[2]) * self.dt
#         # out_F[1, 3] = np.sin(x[2]) * self.dt
#         # out_F[2, 4] = self.dt
#         # out_F[3, 5] = self.dt
#         return out_F

#     def subtract(self, x0: ndarray, x1: ndarray):
#         """Residual function for computing difference in angle"""
#         x0 = self.vec_setter(x0, self.nx)
#         x1 = self.vec_setter(x1, self.nx)
#         xres = x0 - x1
#         xres = xres % (2.0 * np.pi)
#         if xres[2] > np.pi:
#             xres[2] -= 2. * np.pi
#         return xres

#     @ property
#     def labels(self):
#         """Return labels for plotting"""
#         return self._labels


# class CCA(MotionModel):
#     # pylint: disable=invalid-name
#     """Class for Constant Turn Rate and Velocity for a point object"""

#     def __init__(self):
#         super().__init__(nx=6, nq=6, name='CTRA2')
#         self._labels = ['Position X', 'Position Y', 'Heading', 'Speed',
#                         'Curvature', 'Acceleration']

#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None = None,
#         u: ndarray | None = None
#     ) -> ndarray:
#         """Model dynamics function"""
#         x = self.vec_setter(x, self.nx)
#         next_x = deepcopy(x)

#         v = np.sqrt(x[2]**2 + x[3]**2)

#         next_x[0] += x[2] * self.dt + (0.5 * x[5] * x[2] / v) * self.dt**2
#         next_x[1] += x[3] * self.dt + (0.5 * x[5] * x[3] / v) * self.dt**2
#         t1 = x[5] / (v * x[4])
#         next_x[2] += t1 * (x[3] * (np.cos(x[4] * self.dt) - 1)
#                            + x[2] * np.sin(x[4] * self.dt))
#         next_x[3] += -t1 * (x[2] * (np.cos(x[4] * self.dt) - 1)
#                             - x[3] * np.sin(x[4] * self.dt))

#         #
#         # next_x[2] += (x[5] * x[2] / v - x[4] * x[3]) * self.dt
#         # next_x[3] += (x[5] * x[3] / v + x[4] * x[2]) * self.dt
#         # next_x[2] += x[4] * self.dt
#         # next_x[3] += x[5] * self.dt
#         # if next_x[4] > np.pi:
#         #     next_x[4] -= 2 * np.pi
#         # if next_x[4] < -np.pi:
#         #     next_x[4] += 2 * np.pi
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
#         self._sigmas = self.vec_setter(sigmas, 2)

#     def compute_Q(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get Q matrix"""
#         sigma_sq = self.sigmas**2
#         out_Q = np.zeros((self.nx, self.nx))

#         # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
#         # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.

#         # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
#         # out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.

#         # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
#         # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.

#         out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
#         out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
#         return out_Q

#     def compute_G(
#         self,
#         x: ndarray | None = None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get G matrix"""
#         return np.eye(self.nx)

#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None = None
#     ) -> None:
#         """Get F matrix"""
#         out_F = np.eye(self.nx)
#         # out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
#         # out_F[0, 3] = np.cos(x[2]) * self.dt
#         # out_F[1, 2] = x[3] * np.sin(x[2]) * self.dt
#         # out_F[1, 3] = np.sin(x[2]) * self.dt
#         # out_F[2, 4] = self.dt
#         # out_F[3, 5] = self.dt
#         return out_F

#     def subtract(self, x0: ndarray, x1: ndarray):
#         """Residual function for computing difference in angle"""
#         x0 = self.vec_setter(x0, self.nx)
#         x1 = self.vec_setter(x1, self.nx)
#         xres = x0 - x1
#         xres = xres % (2.0 * np.pi)
#         if xres[2] > np.pi:
#             xres[2] -= 2. * np.pi
#         return xres

#     @ property
#     def labels(self):
#         """Return labels for plotting"""
#         return self._labels
