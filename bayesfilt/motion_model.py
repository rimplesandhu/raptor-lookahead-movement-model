"""Base class for defining a motion model"""
from abc import abstractmethod
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
        name: str = 'motion_model'
    ) -> None:

        # model parameters
        super().__init__(nx=nx, name=name)
        self._nq = self.nx if nq is None else self.int_setter(nq)
        self._dt: float  # time interval
        self._qbar = np.zeros((self.nq))

        # model matrices
        self._F: ndarray | None = None  # State transition matrix
        self._G: ndarray | None = None  # Error Jacobian matrix
        self._Q: ndarray | None = None  # Error covariance matrix
        self._labels = [f'x_{i}' for i in range(self.nx)]

    @abstractmethod
    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray
    ) -> None:
        """Update system parameters"""

    @abstractmethod
    def f(
        self,
        x: ndarray,
        q: ndarray | None,
        u: ndarray | None
    ) -> ndarray:
        """Model dynamics equation"""

    @abstractmethod
    def compute_F(
        self,
        x: ndarray | None,
        q: ndarray | None
    ) -> ndarray:
        """Get F matrix"""

    @abstractmethod
    def compute_G(
        self,
        x: ndarray | None,
        q: ndarray | None
    ) -> ndarray:
        """Get G matrix"""

    @abstractmethod
    def compute_Q(
        self,
        x: ndarray | None,
        q: ndarray | None
    ) -> ndarray:
        """Get Q matrix"""

    @property
    def nq(self) -> int:
        """Getter for error dimension"""
        return self._nq

    @property
    def dt(self) -> float:
        """Getter for time interval"""
        return self._dt

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

    @property
    def qbar(self) -> ndarray:
        """Getter for error mean qbar"""
        return self._qbar

    @qbar.setter
    def qbar(self, in_val) -> None:
        """Setter for error mean qbar"""
        self._qbar = self.vec_setter(in_val, self.nq)

    @property
    def labels(self) -> list[str]:
        """Getter for labels"""
        return self._labels

    @labels.setter
    def labels(self, in_list) -> None:
        """Setter for labels"""
        if len(in_list) != self.nx:
            self.raiseit(f'Number of labels should be {self.nx}')
        self._labels = in_list

    def _initiate_matrices_to_identity(self):
        """Initiate all matrices to identity"""
        self._F = np.eye(self._nx)
        self._G = np.eye(self._nx)
        self._Q = np.eye(self._nx)

    def _initiate_matrices_to_zeros(self):
        """Initiate all matrices to zeros"""
        self._F = np.zeros((self._nx, self._nx))
        self._G = np.zeros((self._nx, self._nq))
        self._Q = np.zeros((self._nq, self._nq))

    def __str__(self):
        out_str = f':::{self.name}\n'
        out_str += f'State Dimension: {self._nx}\n'
        out_str += f'Error Dimension: {self._nq}\n'
        out_str += f'F:\n {np.array_str(np.array(self.F), precision=3)}\n'
        out_str += f'G:\n {np.array_str(np.array(self.G), precision=3)}\n'
        out_str += f'Q:\n {np.array_str(np.array(self.Q), precision=3)}\n'
        return out_str
