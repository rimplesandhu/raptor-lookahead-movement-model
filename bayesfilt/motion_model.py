"""Base class for defining a motion model"""
import numpy as np
from numpy import ndarray
from .state_space_model import StateSpaceModel


class MotionModel(StateSpaceModel):
    # pylint: disable=invalid-name
    """Base class for defining a motion model"""

    def __init__(
        self,
        nx: int,
        nq: int | None = None,
        name: str = 'MotionModel'
    ) -> None:

        super().__init__(nx=nx, name=name)
        self._nq = self.nx if nq is None else self.int_setter(nq)
        self._dt: float | None = None  # time interval
        self._F: ndarray | None = None  # State transition matrix
        self._G: ndarray | None = None  # Error Jacobian matrix
        self._Q: ndarray | None = None  # Error covariance matrix
        self._qbar: ndarray | None = None  # Error mean vector

    def subtract_states(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference among states"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        return np.subtract(x0, x1)

    def _check_if_model_initiated_correctly(self):
        """Checks if all the model parameters are initiated"""
        for key, val in self.phi.items():
            if val is None:
                self.raiseit(f'Parameter {key} not assigned!')
        assert self.dt is not None, 'Need to assign dt'

    @property
    def qbar(self) -> float:
        """Getter for time interval"""
        return self._qbar

    @qbar.setter
    def qbar(self, in_list) -> float:
        """Getter for time interval"""
        if len(in_list) != self.nx:
            self.raiseit(f'Dimension of vector u should be {self.nx}')
        self._qbar = in_list

    @property
    def nq(self) -> int:
        """Getter for error dimension"""
        return self._nq

    @property
    def dt(self) -> float:
        """Getter for time interval"""
        return self._dt

    @dt.setter
    def dt(self, in_val: float) -> float:
        """Getter for time interval"""
        self._dt = self.float_setter(in_val)

    @property
    def F(self) -> ndarray:
        """Getter for process matrix F"""
        return self._F

    @property
    def G(self) -> ndarray:
        """Getter for error jacobian matrix G"""
        return self._G

    @property
    def Q(self) -> ndarray:
        """Getter for error covariance matrix Q"""
        return self._Q

    def _initiate_matrices_to_identity(self):
        """Initiate all matrices to identity"""
        self._F = np.eye(self.nx)
        self._G = np.eye(self.nx)
        self._Q = np.eye(self.nx)
        self.qbar = np.zeros((self.nx,))

    def _initiate_matrices_to_zeros(self):
        """Initiate all matrices to zeros"""
        self._F = np.zeros((self.nx, self.nx))
        self._G = np.zeros((self.nx, self.nq))
        self._Q = np.zeros((self.nq, self.nq))
        self.qbar = np.zeros((self.nx,))
