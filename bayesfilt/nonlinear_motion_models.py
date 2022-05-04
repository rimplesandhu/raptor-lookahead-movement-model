"""Classes defining nonlinear motion models"""
from abc import abstractmethod
import numpy as np
from numpy import ndarray
from .motion_model import MotionModel


class CTRV2D(MotionModel):
    # pylint: disable=invalid-name
    """Class for Constant Turn Rate and Velocity in 2D"""

    def __init__(self):
        super().__init__(nx=5, nq=5, name='CTRV2D')

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics function"""
        x = self.vec_setter(x, self.nx)
        next_x = x.copy()
        next_x[2] += x[4] * self.dt
        next_x[0] = x[0] + x[3] / x[4] * (np.sin(next_x[2]) - np.sin(x[2]))
        next_x[1] = x[1] - x[3] / x[4] * (np.cos(next_x[2]) - np.cos(x[2]))
        next_x[3] = x[3]
        next_x[4] = x[4]
        if q is not None:
            q = self.vec_setter(q, self.nx)
            next_x += q
        if u is not None:
            u = self.vec_setter(q, self.nx)
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

    def get_F(
        self,
        x: ndarray | None,
        q: ndarray | None = None
    ) -> None:
        """Get F matrix"""

        # common terms
        atmp = x[3] / x[4] * (np.sin(x[2] + x[4] * self.dt) - np.sin(x[2]))
        btmp = x[3] / x[4] * (np.cos(x[2] + x[4] * self.dt) - np.cos(x[2]))
        ctmp = (x[3] * self.dt / x[4]) * np.sin(x[4] * self.dt + x[2])
        dtmp = (x[3] * self.dt / x[4]) * np.cos(x[4] * self.dt + x[2])

        # Jacobian matrix F
        self._F = np.eye(self.nx)
        self._F[0, 2] = btmp
        self._F[0, 3] = atmp / x[3]
        self._F[0, 4] = -atmp / x[4] + dtmp
        self._F[1, 2] = atmp
        self._F[1, 3] = -btmp / x[3]
        self._F[1, 4] = btmp / x[4] + ctmp
        self._F[2, 4] = self.dt
        return self._F

    def get_Q(
        self,
        x: ndarray | None,
        q: ndarray | None = None
    ) -> None:
        """Get Q matrix"""
        fac = 1.
        sigma_sq = self.sigmas**2
        self._Q = np.eye(self.nx)
        self._Q[0, 0] = sigma_sq[0] * np.cos(x[2]) ** 2 * self.dt ** 3 / 3.
        self._Q[0, 1] = sigma_sq[0] * np.sin(2 * x[2]) * self.dt ** 3 / 6.
        self._Q[0, 3] = sigma_sq[0] * np.cos(x[2]) * self.dt ** 2 / 2.
        self._Q[0, 4] = sigma_sq[1] * x[3] * np.sin(x[2]) * self.dt ** 3 / 6.
        self._Q[1, 1] = sigma_sq[0] * np.sin(x[2]) ** 2 * self.dt ** 3 / 3.
        self._Q[1, 3] = sigma_sq[0] * np.sin(x[2]) * self.dt ** 2 / 2.
        self._Q[1, 4] = sigma_sq[1] * x[3] * np.cos(x[2]) * self.dt ** 3 / 6.
        self._Q[2, 2] = sigma_sq[1] * self.dt ** 3 / 3.
        self._Q[2, 4] = sigma_sq[1] * self.dt ** 2 / 2.
        self._Q[3, 3] = sigma_sq[0] * self.dt ** 2 / 1.
        self._Q[4, 4] = sigma_sq[1] * self.dt ** 2 / 1.

    def get_G(
        self,
        x: ndarray | None,
        q: ndarray | None
    ) -> None:
        """Get G matrix"""
        self._G = np.eye(self.nx)
        return self._G
