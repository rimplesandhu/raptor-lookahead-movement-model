""" Classes for defining linear motion models """
from abc import abstractmethod
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
    def compute(
        self,
        delta_t: float,
        model_error_stds: ndarray,
        gamma_par: float
    ) -> None:
        pass

    @property
    def dim(self) -> int:
        """Returns dimension (size) of the state vector"""
        return self._dim

    @property
    def F(self) -> ndarray:
        """Returns process matrix F"""
        return self._F

    @property
    def Q(self) -> ndarray:
        """Returns process error covariance matrix Q"""
        return self._Q

    def __str__(self):
        out_str = ':::Linear motion model\n'
        out_str += f'Dimension: {self._dim}\n'
        out_str += f'Process mat F:\n {np.array_str(self._F, precision=3)}\n'
        out_str += f'Error cov mat Q:\n {np.array_str(self._Q, precision=3)}\n'
        return out_str


class RandomWalk1D(LinearMotionModel):
    """Class for random walk 1d model """

    def __init__(self):
        super().__init__(dim=1)

    def compute(
        self,
        delta_t: float,
        model_error_stds: float,
        gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        self._F[0, 0] = 1.
        self._Q[0, 0] = 1. * delta_t**1 / 1
        model_error_stds = np.asarray(model_error_stds)
        assert model_error_stds.size == 1, 'Need 1 standard deviation for RW1D!'
        self._Q *= model_error_stds**2


class RandomWalkND(LinearMotionModel):
    """ Class for random walk model in ND """

    def __init__(self, dof: int):
        super().__init__(dim=dof)
        self._rw1d = RandomWalk1D()

    def compute(
        self,
        delta_t: float,
        model_error_stds: ndarray,
        gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        model_error_stds = np.asarray(model_error_stds)
        out_string = 'Size mismatch for error standard deviations!'
        assert model_error_stds.size == self.dim, out_string
        for i, error_std in enumerate(model_error_stds):
            self._rw1d.compute(delta_t, error_std, gamma_par=gamma_par)
            self._F[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.F.copy()
            self._Q[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.Q.copy()


class ConstantVelocity1D(LinearMotionModel):
    """ Class for constant velocity model in 1D """

    def __init__(self):
        super().__init__(dim=2)

    def compute(
        self,
        delta_t: float,
        model_error_stds: float,
        gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        self._F[0, 1] = delta_t
        self._F[1, 1] = gamma_par
        self._Q[0, 0] = 1. * delta_t**3 / 3
        self._Q[0, 1] = 1. * delta_t**2 / 2
        self._Q[1, 1] = delta_t
        self._Q = symmetrize(self._Q)
        model_error_stds = np.asarray(model_error_stds)
        assert model_error_stds.size == 1, 'Need 1 standard deviation for CV1D!'
        self._Q *= model_error_stds**2


class ConstantVelocityND(LinearMotionModel):
    """Class for constant velocity model in N-dimensions"""

    def __init__(self, dof: int):
        super().__init__(dim=2 * dof)
        self._cv1d = ConstantVelocity1D()

    def compute(
        self,
        delta_t: float,
        model_error_stds: ndarray,
        gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        model_error_stds = np.asarray(model_error_stds)
        out_string = 'Size mismatch for error standard deviations!'
        assert model_error_stds.size == int(self.dim / 2), out_string
        for i, error_std in enumerate(model_error_stds):
            self._cv1d.compute(delta_t, error_std, gamma_par=gamma_par)
            self._F[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.F.copy()
            self._Q[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.Q.copy()


class ConstantAcceleration1D(LinearMotionModel):
    """ Class for constant acceleration model in 1D """

    def __init__(self):
        super().__init__(dim=3)

    def compute(
            self,
            delta_t: float,
            model_error_stds: float,
            gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        self._F[0, 1] = delta_t
        self._F[0, 2] = delta_t**2 / 2
        self._F[1, 2] = gamma_par * delta_t
        self._Q[0, 0] = 1. * delta_t**5 / 20
        self._Q[0, 1] = 1 * delta_t**4 / 8
        self._Q[0, 2] = 1 * delta_t**3 / 6
        self._Q[1, 1] = 1. * delta_t**3 / 3
        self._Q[1, 2] = 1 * delta_t**2 / 2
        self._Q[2, 2] = delta_t
        self._Q = symmetrize(self.Q)
        model_error_stds = np.asarray(model_error_stds)
        assert model_error_stds.size == 1, 'Need 1 standard deviation for CA1D!'
        self._Q *= model_error_stds**2


class ConstantAccelerationND(LinearMotionModel):
    """ Class for constant acceleration model in N-dimension"""

    def __init__(self, dof: int):
        super().__init__(dim=3 * dof)
        self._ca1d = ConstantAcceleration1D()

    def compute(
        self,
        delta_t: float,
        model_error_stds: ndarray,
        gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        model_error_stds = np.asarray(model_error_stds)
        out_string = 'Size mismatch for error standard deviations!'
        assert model_error_stds.size == int(self.dim / 3), out_string
        for i, error_std in enumerate(model_error_stds):
            self._ca1d.compute(delta_t, error_std, gamma_par=gamma_par)
            self._F[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.F.copy()
            self._Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.Q.copy()
