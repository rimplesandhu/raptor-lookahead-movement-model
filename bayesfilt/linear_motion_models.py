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


class ConstantVelocity2D(LinearMotionModel):
    """ Class for constant velocity model in 2D """

    def __init__(self):
        super().__init__(dim=4)
        self._cv1d = ConstantVelocity1D()

    def compute(self, dt: float, q_std_vec: ndarray,
                gamma_par: float = 1.) -> None:
        """ Computes updated model matrices """
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == 2, 'Need 2 standard deviations for CV2D!'
        self._cv1d.compute(dt, q_std_vec[0], gamma_par=gamma_par)
        cv1d_fmat_1 = self._cv1d.F.copy()
        cv1d_qmat_1 = self._cv1d.Q.copy()
        self._cv1d.compute(dt, q_std_vec[1], gamma_par=gamma_par)
        cv1d_fmat_2 = self._cv1d.F.copy()
        cv1d_qmat_2 = self._cv1d.Q.copy()
        zero_mat = np.zeros_like(cv1d_fmat_1)
        self._F = np.block([[cv1d_fmat_1, zero_mat],
                            [zero_mat, cv1d_fmat_2]])
        self._Q = np.block([[cv1d_qmat_1, zero_mat],
                            [zero_mat, cv1d_qmat_2]])


class ConstantVelocity3D(LinearMotionModel):
    """ Class for constant velocity model in 3D """

    def __init__(self):
        super().__init__(dim=6)
        self._cv1d_1 = ConstantVelocity1D()
        self._cv1d_2 = ConstantVelocity1D()
        self._cv1d_3 = ConstantVelocity1D()

    def compute(self, dt: float, q_std_vec: ndarray,
                gamma_par: float = 1.0) -> None:
        """ Computes updated model matrices """
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == 3, 'Need 3 standard deviations for CV3D!'
        self._cv1d_1.compute(dt, q_std_vec[0], gamma_par=gamma_par)
        cv1d_fmat_1 = self._cv1d_1.F
        cv1d_qmat_1 = self._cv1d_1.Q
        self._cv1d_2.compute(dt, q_std_vec[1], gamma_par=gamma_par)
        cv1d_fmat_2 = self._cv1d_2.F
        cv1d_qmat_2 = self._cv1d_2.Q
        self._cv1d_3.compute(dt, q_std_vec[2], gamma_par=gamma_par)
        cv1d_fmat_3 = self._cv1d_3.F
        cv1d_qmat_3 = self._cv1d_3.Q
        zero_mat = np.zeros_like(cv1d_fmat_1)
        self._F = np.block([[cv1d_fmat_1, zero_mat, zero_mat],
                            [zero_mat, cv1d_fmat_2, zero_mat],
                            [zero_mat, zero_mat, cv1d_fmat_3]
                            ]).copy()
        self._Q = np.block([[cv1d_qmat_1, zero_mat, zero_mat],
                            [zero_mat, cv1d_qmat_2, zero_mat],
                            [zero_mat, zero_mat, cv1d_qmat_3]
                            ]).copy()


class ConstantAcceleration1D(LinearMotionModel):
    """ Class for constant acceleration model in 1D """

    def __init__(self):
        super().__init__(dim=3)

    def compute(self, dt: float, q_std_vec: float) -> None:
        """ Computes updated model matrices """
        self._F[0, 1] = dt
        self._F[0, 2] = dt**2 / 2
        self._F[1, 2] = dt
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


class ConstantAcceleration2D(LinearMotionModel):
    """ Class for constant acceleration model in 2D """

    def __init__(self):
        super().__init__(dim=6)
        self._ca1d = ConstantAcceleration1D()

    def compute(self, dt: float, q_std_vec: ndarray) -> None:
        """ returns F matrix """
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == 2, 'Need 2 standard deviations for CA2D!'
        self._ca1d.compute(dt, q_std_vec[0])
        ca1d_fmat_1 = self._ca1d.F
        ca1d_qmat_1 = self._ca1d.Q
        self._ca1d.compute(dt, q_std_vec[1])
        ca1d_fmat_2 = self._ca1d.F
        ca1d_qmat_2 = self._ca1d.Q
        zero_mat = np.zeros_like(ca1d_fmat_1)
        self._F = np.block([[ca1d_fmat_1, zero_mat],
                            [zero_mat, ca1d_fmat_2]])
        self._Q = np.block([[ca1d_qmat_1, zero_mat],
                            [zero_mat, ca1d_qmat_2]])


class ConstantAcceleration3D(LinearMotionModel):
    """ Class for constant acceleration model in 3D """

    def __init__(self):
        super().__init__(dim=9)
        self._ca1d_1 = ConstantAcceleration1D()
        self._ca1d_2 = ConstantAcceleration1D()
        self._ca1d_3 = ConstantAcceleration1D()

    def compute(self, dt: float, q_std_vec: ndarray) -> None:
        """ Computes updated model matrices """
        q_std_vec = np.asarray(q_std_vec)
        assert q_std_vec.size == 3, 'Need 3 standard deviations for CA3D!'
        self._ca1d_1.compute(dt, q_std_vec[0])
        ca1d_fmat_1 = self._ca1d_1.F
        ca1d_qmat_1 = self._ca1d_1.Q
        self._ca1d_2.compute(dt, q_std_vec[1])
        ca1d_fmat_2 = self._ca1d_2.F
        ca1d_qmat_2 = self._ca1d_2.Q
        self._ca1d_3.compute(dt, q_std_vec[2])
        ca1d_fmat_3 = self._ca1d_3.F
        ca1d_qmat_3 = self._ca1d_3.Q
        zero_mat = np.zeros_like(ca1d_fmat_1)
        self._F = np.block([[ca1d_fmat_1, zero_mat, zero_mat],
                            [zero_mat, ca1d_fmat_2, zero_mat],
                            [zero_mat, zero_mat, ca1d_fmat_3]
                            ]).copy()
        self._Q = np.block([[ca1d_qmat_1, zero_mat, zero_mat],
                            [zero_mat, ca1d_qmat_2, zero_mat],
                            [zero_mat, zero_mat, ca1d_qmat_3]
                            ]).copy()
