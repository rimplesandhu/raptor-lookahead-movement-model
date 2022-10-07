"""Classes defining Constant Curvature and Acceleration Model"""
from copy import deepcopy
import numpy as np
from numpy import ndarray
from .motion_model import MotionModel


class CCA(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=6, nq=6, name='CCA')
        self._labels = ['Position X', 'Position Y', 'Heading', 'Speed',
                        'Acceleration', 'Rcurvature']

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        next_x[3] += x[4] * self.dt  # speed
        if abs(x[5]) < 20000.:
            next_x[2] += (next_x[3]**2 - x[3]**2) / (2. * x[4] * x[5])
            next_x[1] += x[5] * (np.sin(next_x[2]) - np.sin(x[2]))  # x
            next_x[0] += -x[5] * (np.cos(next_x[2]) - np.cos(x[2]))  # y
        else:
            next_x[1] += (next_x[3]**2 - x[3]**2) * np.cos(x[2]) / (2. * x[4])
            next_x[0] += (next_x[3]**2 - x[3]**2) * np.sin(x[2]) / (2. * x[4])
        # if next_x[2] > np.pi:
        #     next_x[2] -= 2.0 * np.pi
        # next_x[3] += x[4] * self.dt  # speed

        if q is not None:
            q = self.vec_setter(q, self.nx)
            next_x += q
        if u is not None:
            u = self.vec_setter(u, self.nx)
            next_x += u
        return next_x

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters and matrices"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.vec_setter(sigmas, 2)

    def compute_F(
        self,
        x: ndarray | None,
        q: ndarray | None = None
    ) -> None:
        """Get F matrix"""
        out_F = np.eye(self.nx)
        return out_F

    def compute_Q(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get Q matrix"""
        sigma_sq = self.sigmas**2
        out_Q = np.zeros((self.nx, self.nx))
        out_Q[2, 2] = (10 * np.pi / 180)**2
        out_Q[3, 3] = sigma_sq[0] * self.dt ** 3 / 3.
        out_Q[3, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 3] = sigma_sq[0] * self.dt ** 2 / 2.
        # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
        out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
        return out_Q

    def compute_G(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get G matrix"""
        return np.eye(self.nx)

    def subtract(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference in angle"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        xres = x0 - x1
        xres[2] = xres[2] % (2.0 * np.pi)
        if xres[2] > np.pi:
            xres[2] -= 2. * np.pi
        return xres

    @ property
    def labels(self):
        """Return labels for plotting"""
        return ['Position X', 'Position Y', 'Heading', 'Speed',
                'Acceleration', 'Rcurv']


class CCAold(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=6, nq=6, name='CTRA')
        self._labels = ['Position X', 'Position Y', 'Heading', 'Speed',
                        'Acceleration', 'Curvature']

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)

        next_x[2] += -x[5] * (x[3] * self.dt +
                              0.5 * x[4] * self.dt**2)  # yaw
        next_x[2] = next_x[2] % (2.0 * np.pi)
        if next_x[2] > np.pi:
            next_x[2] -= 2.0 * np.pi
        next_x[3] += x[4] * self.dt  # speed

        next_x[0] += next_x[3] * np.cos(x[2] - x[5] * next_x[3] * self.dt)  # x
        next_x[1] += next_x[3] * np.sin(x[2] - x[5] * next_x[3] * self.dt)  # y

        # next_x[0] += abs(next_x[3]) * np.cos(next_x[2])  # x
        # ext_x[1] += abs(next_x[3]) * np.sin(next_x[2])  # y

        # next_x[0] += abs(x[3]) * np.cos(x[2])  # x
        # next_x[1] += abs(x[3]) * np.sin(x[2])  # y

        if q is not None:
            q = self.vec_setter(q, self.nx)
            next_x += q
        if u is not None:
            u = self.vec_setter(u, self.nx)
            next_x += u
        return next_x

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters and matrices"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.vec_setter(sigmas, 2)

    def compute_F(
        self,
        x: ndarray | None,
        q: ndarray | None = None
    ) -> None:
        """Get F matrix"""
        out_F = np.eye(self.nx)
        return out_F

    def compute_Q(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get Q matrix"""
        sigma_sq = self.sigmas**2
        out_Q = np.zeros((self.nx, self.nx))
        # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
        # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
        # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
        # out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.
        # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
        out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
        return out_Q

    def compute_G(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get G matrix"""
        return np.eye(self.nx)

    def subtract(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference in angle"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        xres = x0 - x1
        xres[2] = xres[2] % (2.0 * np.pi)
        if xres[2] > np.pi:
            xres[2] -= 2. * np.pi
        return xres

    @ property
    def labels(self):
        """Return labels for plotting"""
        return ['Position X', 'Position Y', 'Heading', 'Speed',
                'Angular Speed', 'Acceleration']


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


class CTRA2(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=6, nq=6, name='CTRA2')
        self._labels = ['Position X', 'Position Y', 'SpeedX', 'SpeedY',
                        'Angular speed', 'Acceleration']

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)

        v = np.sqrt(x[2]**2 + x[3]**2)
        next_x[0] += x[2] * self.dt + (0.5 * x[5] * x[2] / v) * self.dt**2
        next_x[1] += x[3] * self.dt + (0.5 * x[5] * x[3] / v) * self.dt**2
        t1 = x[5] / (v * x[4])
        next_x[2] += t1 * (x[3] * (np.cos(x[4] * self.dt) - 1)
                           + x[2] * np.sin(x[4] * self.dt))
        next_x[3] += -t1 * (x[2] * (np.cos(x[4] * self.dt) - 1)
                            - x[3] * np.sin(x[4] * self.dt))

        #
        # next_x[2] += (x[5] * x[2] / v - x[4] * x[3]) * self.dt
        # next_x[3] += (x[5] * x[3] / v + x[4] * x[2]) * self.dt
        # next_x[2] += x[4] * self.dt
        # next_x[3] += x[5] * self.dt
        # if next_x[4] > np.pi:
        #     next_x[4] -= 2 * np.pi
        # if next_x[4] < -np.pi:
        #     next_x[4] += 2 * np.pi
        if q is not None:
            q = self.vec_setter(q, self.nx)
            next_x += q
        if u is not None:
            u = self.vec_setter(u, self.nx)
            next_x += u
        return next_x

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters and matrices"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.vec_setter(sigmas, 2)

    def compute_Q(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get Q matrix"""
        sigma_sq = self.sigmas**2
        out_Q = np.zeros((self.nx, self.nx))

        # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
        # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.

        # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
        # out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.

        # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        # out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.

        out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.
        out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.
        return out_Q

    def compute_G(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get G matrix"""
        return np.eye(self.nx)

    def compute_F(
        self,
        x: ndarray | None,
        q: ndarray | None = None
    ) -> None:
        """Get F matrix"""
        out_F = np.eye(self.nx)
        # out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
        # out_F[0, 3] = np.cos(x[2]) * self.dt
        # out_F[1, 2] = x[3] * np.sin(x[2]) * self.dt
        # out_F[1, 3] = np.sin(x[2]) * self.dt
        # out_F[2, 4] = self.dt
        # out_F[3, 5] = self.dt
        return out_F

    def subtract(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference in angle"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        xres = x0 - x1
        xres = xres % (2.0 * np.pi)
        if xres[2] > np.pi:
            xres[2] -= 2. * np.pi
        return xres

    @ property
    def labels(self):
        """Return labels for plotting"""
        return self._labels


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
