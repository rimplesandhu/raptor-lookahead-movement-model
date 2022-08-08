"""Classes defining Constant Turn Rate and Velocity Models"""
from copy import deepcopy
import numpy as np
from numpy import ndarray
from .motion_model import MotionModel


class CTRA2D(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity in 2D"""

    def __init__(self):
        super().__init__(nx=6, nq=6, name='CTRA2D')
        self.labels = ['X', 'Y', 'Heading', 'Hspeed', 'HeadingRate', 'Haccn']

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        next_x[4] = min(abs(next_x[4]), np.pi / 7.) * \
            (next_x[4] / abs(next_x[4]))

        next_x[3] += x[5] * self.dt

        #next_x[3] = abs(next_x[3])
        # next_x[0] += x[3] * np.sin(x[2]) * self.dt
        # next_x[1] += x[3] * np.cos(x[2]) * self.dt
        #     next_x[0] += x[3] * np.sin(x[2]) * self.dt
        #     next_x[1] += x[3] * np.cos(x[2]) * self.dt
        # else:
        # if abs(x[4]) > np.pi / 4:
        next_x[2] += x[4] * self.dt
        next_x[2] = next_x[2] % (2.0 * np.pi)
        if next_x[2] > np.pi:
            next_x[2] -= 2. * np.pi
        t1 = 1 / (x[4]**2)
        next_x[1] += t1 * (next_x[3] * x[4] * np.sin(next_x[2])
                           - x[3] * x[4] * np.sin(x[2])
                           + x[5] * np.cos(next_x[2])
                           - x[5] * np.cos(x[2]))

        next_x[0] += t1 * (-next_x[3] * x[4] * np.cos(next_x[2])
                           + x[3] * x[4] * np.cos(x[2])
                           + x[5] * np.sin(next_x[2])
                           - x[5] * np.sin(x[2]))

        #curv = next_x[4] / next_x[3]

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

        # common terms
        # atmp = x[3] / x[4] * (np.sin(x[2] + x[4] * self.dt) - np.sin(x[2]))
        # btmp = x[3] / x[4] * (np.cos(x[2] + x[4] * self.dt) - np.cos(x[2]))
        # ctmp = (x[3] * self.dt / x[4]) * np.sin(x[4] * self.dt + x[2])
        # dtmp = (x[3] * self.dt / x[4]) * np.cos(x[4] * self.dt + x[2])
        # Jacobian matrix F
        out_F = np.eye(self.nx)
        out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
        out_F[0, 3] = np.cos(x[2]) * self.dt
        out_F[1, 2] = x[3] * np.cos(x[2]) * self.dt
        out_F[1, 3] = np.sin(x[2]) * self.dt
        out_F[2, 4] = self.dt
        out_F[3, 5] = self.dt
        return out_F

    def compute_Q(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get Q matrix"""
        sigma_sq = self.sigmas**2
        out_Q = np.zeros((self.nx, self.nx))
        # if x is not None:
        #     out_Q[0, 0] = sigma_sq[1] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
        #     out_Q[0, 1] = sigma_sq[1] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
        #     out_Q[0, 3] = sigma_sq[1] * np.cos(x[2]) * self.dt ** 2 / 2.
        #     out_Q[0, 4] = sigma_sq[0] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
        #     out_Q[1, 1] = sigma_sq[1] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
        #     out_Q[1, 3] = sigma_sq[1] * np.sin(x[2]) * self.dt ** 2 / 2.
        #     out_Q[1, 4] = sigma_sq[0] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
        # out_Q[0, 0] = -sigma_sq[0] * self.dt ** 3 / 3.
        # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
        # # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
        # # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
        #
        out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
        out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.

        out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
        out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
        out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.
        out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.

        return out_Q

    def subtract(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference among states"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        xres = x0 - x1
        xres[2] = xres[2] % (2.0 * np.pi)
        if xres[2] > np.pi:
            xres[2] -= 2. * np.pi
        return xres

    def compute_G(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get G matrix"""
        return np.eye(self.nx)


class CTRA3D(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity for a point object"""

    def __init__(self):
        super().__init__(nx=9, nq=9, name='CTRA')
        self.labels = ['X', 'Y', 'Heading', 'Hspeed', 'HeadingRate', 'Haccn',
                       'Altitude', 'Vspeed', 'Vaccn']

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        x = self.vec_setter(x, self.nx)
        next_x = deepcopy(x)
        next_x[2] += x[4] * self.dt
        next_x[3] += x[5] * self.dt
        next_x[0] += x[3] * np.sin(x[2]) * self.dt
        next_x[1] += x[3] * np.cos(x[2]) * self.dt
        #     t1 = 1 / (x[4]**2)
        #     next_x[1] += t1 * (next_x[3] * x[4] * np.sin(next_x[2])
        #                     - x[3] * x[4] * np.sin(x[2])
        #                     + x[5] * np.cos(next_x[2])
        #                     - x[5] * np.cos(x[2]))

        #     next_x[0] += t1 * (-next_x[3] * x[4] * np.cos(next_x[2])
        #                     + x[3] * x[4] * np.cos(x[2])
        #                     + x[5] * np.sin(next_x[2])
        #                     - x[5] * np.sin(x[2]))

        next_x[6] += x[7] * self.dt + 0.5 * x[8] * self.dt**2
        next_x[7] += x[8] * self.dt

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
        self._sigmas = self.vec_setter(sigmas, 3)

    def compute_F(
        self,
        x: ndarray | None,
        q: ndarray | None = None
    ) -> None:
        """Get F matrix"""
        # atmp = x[3] / x[4] * (np.sin(x[2] + x[4] * self.dt) - np.sin(x[2]))
        # btmp = x[3] / x[4] * (np.cos(x[2] + x[4] * self.dt) - np.cos(x[2]))
        # ctmp = (x[3] * self.dt / x[4]) * np.sin(x[4] * self.dt + x[2])
        # dtmp = (x[3] * self.dt / x[4]) * np.cos(x[4] * self.dt + x[2])
        out_F = np.eye(self.nx)
        out_F[0, 2] = -x[3] * np.sin(x[2]) * self.dt
        out_F[0, 3] = np.cos(x[2]) * self.dt
        out_F[1, 2] = x[3] * np.cos(x[2]) * self.dt
        out_F[1, 3] = np.sin(x[2]) * self.dt
        out_F[2, 4] = self.dt
        out_F[3, 5] = self.dt
        return out_F

    def compute_Q(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get Q matrix"""
        sigma_sq = self.sigmas**2
        out_Q = np.zeros((self.nx, self.nx))
        # if x is not None:
        #     out_Q[0, 0] = sigma_sq[1] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
        #     out_Q[0, 1] = sigma_sq[1] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
        #     out_Q[0, 3] = sigma_sq[1] * np.cos(x[2]) * self.dt ** 2 / 2.
        #     out_Q[0, 4] = sigma_sq[0] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
        #     out_Q[1, 1] = sigma_sq[1] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
        #     out_Q[1, 3] = sigma_sq[1] * np.sin(x[2]) * self.dt ** 2 / 2.
        #     out_Q[1, 4] = sigma_sq[0] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
        # out_Q[0, 0] = -sigma_sq[0] * self.dt ** 3 / 3.
        # out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
        # # out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        # out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
        # # out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
        #

        out_Q[2, 2] = sigma_sq[0] * self.dt ** 3 / 3.
        out_Q[2, 4] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 2] = sigma_sq[0] * self.dt ** 2 / 2.
        out_Q[4, 4] = sigma_sq[0] * self.dt ** 1 / 1.

        out_Q[3, 3] = sigma_sq[1] * self.dt ** 3 / 3.
        out_Q[3, 5] = sigma_sq[1] * self.dt ** 2 / 2.
        out_Q[5, 3] = sigma_sq[1] * self.dt ** 2 / 2.
        out_Q[5, 5] = sigma_sq[1] * self.dt ** 1 / 1.

        out_Q[6, 6] = sigma_sq[2] * self.dt**5 / 20
        out_Q[6, 7] = sigma_sq[2] * self.dt**4 / 8
        out_Q[7, 6] = sigma_sq[2] * self.dt**4 / 8
        out_Q[6, 8] = sigma_sq[2] * self.dt**3 / 6
        out_Q[8, 6] = sigma_sq[2] * self.dt**3 / 6
        out_Q[7, 7] = sigma_sq[2] * self.dt**3 / 3
        out_Q[7, 8] = sigma_sq[2] * self.dt**2 / 2
        out_Q[8, 7] = sigma_sq[2] * self.dt**2 / 2
        out_Q[8, 8] = sigma_sq[2] * self.dt**1 / 1
        return out_Q

    def subtract(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference among states"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        xres = x0 - x1
        xres[2] = xres[2] % (2.0 * np.pi)
        if xres[2] > np.pi:
            xres[2] -= 2. * np.pi
        return xres

    def compute_G(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> None:
        """Get G matrix"""
        return np.eye(self.nx)
