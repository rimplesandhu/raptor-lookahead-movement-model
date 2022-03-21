""" Classes for defining linear motion models """
from abc import abstractmethod
from typing import Tuple
import numpy as np
from numpy import ndarray


def symmetrize(a_mat: np.ndarray):
    """ Return a symmetrized version of NumPy array """
    return a_mat + a_mat.T - np.diag(a_mat.diagonal())


class LinearMotionModel:
    """ Base class for defining a linear motion model with additive
    Gaussain errors"""

    def __init__(self, dim: int) -> None:
        self._dim: int = dim  # dimension
        self._F: ndarray = np.eye(self.dim)  # State transition matrix
        self._Q: ndarray = np.eye(self.dim)  # model error covariance matrix

    @abstractmethod
    def compute(self, dt: float, q_std_vec: ndarray, gamma_par: float) -> None:
        pass

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def F(self) -> ndarray:
        return self._F

    @property
    def Q(self) -> ndarray:
        return self._Q


class ConstantVelocity1D(LinearMotionModel):
    """ Class for constant velocity model in 1D """

    def __init__(self):
        super().__init__(dim=2)

    def compute(self, dt: float, q_std_vec: float,
                gamma_par: float = 1.) -> None:
        """ Computes updated model matrices """
        self._F[0, 1] = dt
        self._F[1, 1] = gamma_par
        self._Q[0, 0] = 1. * dt**3 / 3
        self._Q[0, 1] = 1. * dt**2 / 2
        self._Q[1, 1] = dt
        self._Q = symmetrize(self._Q)
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == 1, 'Need 1 standard deviation for CV1D!'
        self._Q *= q_std_vec**2


class ConstantVelocityND(LinearMotionModel):
    """ Class for constant velocity model in 2D """

    def __init__(self, in_dim: int):
        super().__init__(dim=2 * in_dim)
        self._cv1d = ConstantVelocity1D()

    def compute(self, dt: float, q_std_vec: ndarray,
                gamma_par: float = 1.) -> None:
        """ Computes updated model matrices """
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == self.dim / 2, 'Size mismatch for error stds!'
        for i, q_std in enumerate(q_std_vec):
            self._cv1d.compute(dt, q_std, gamma_par=gamma_par)
            self._F[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.F.copy()
            self._Q[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.Q.copy()


class ConstantAcceleration1D(LinearMotionModel):
    """ Class for constant acceleration model in 1D """

    def __init__(self):
        super().__init__(dim=3)

    def compute(self, dt: float, q_std_vec: float,
                gamma_par: float = 1.) -> None:
        """ Computes updated model matrices """
        self._F[0, 1] = dt
        self._F[0, 2] = dt**2 / 2
        self._F[1, 2] = gamma_par * dt
        self._Q[0, 0] = 1. * dt**5 / 20
        self._Q[0, 1] = 1 * dt**4 / 8
        self._Q[0, 2] = 1 * dt**3 / 6
        self._Q[1, 1] = 1. * dt**3 / 3
        self._Q[1, 2] = 1 * dt**2 / 2
        self._Q[2, 2] = dt
        self._Q = symmetrize(self.Q)
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == 1, 'Need 1 standard deviation for CA1D!'
        self._Q *= q_std_vec**2


class ConstantAccelerationND(LinearMotionModel):
    """ Class for constant velocity model in 2D """

    def __init__(self, in_dim: int):
        super().__init__(dim=3 * in_dim)
        self._ca1d = ConstantAcceleration1D()

    def compute(self, dt: float, q_std_vec: ndarray,
                gamma_par: float = 1.) -> None:
        """ Computes updated model matrices """
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == self.dim / 3, 'Size mismatch for error stds!'
        for i, q_std in enumerate(q_std_vec):
            self._ca1d.compute(dt, q_std, gamma_par=gamma_par)
            self._F[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.F.copy()
            self._Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.Q.copy()
