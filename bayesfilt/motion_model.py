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
        name: str = 'MotionModel'
    ) -> None:

        super().__init__(nx=nx, name=name)
        self._dt: float | None = None  # time interval
        self._F: ndarray = np.zeros((self.nx, self.nx))  # Transition matrix
        self._Q: ndarray = np.zeros((self.nx, self.nx))  # Covariance matrix
        self._G: ndarray = np.eye(self.nx)  # Error Jacobian matrix
        self._qbar: ndarray = np.zeros((self.nx,))  # Error mean vector

    def __str__(self):
        out_str = StateSpaceModel.__str__(self)
        out_str += f'dt: {self.dt} second\n'
        out_str += f'qbar:\n {np.array_str(np.array(self.qbar), precision=3)}\n'
        out_str += f'F:\n {np.array_str(np.array(self.F), precision=3)}\n'
        out_str += f'G:\n {np.array_str(np.array(self.G), precision=3)}\n'
        out_str += f'Q:\n {np.array_str(np.array(self.Q), precision=3)}\n'
        return out_str

# functions

    def subtract_states(self, x0: ndarray, x1: ndarray):
        """Residual function for computing difference among states"""
        x0 = self.vec_setter(x0, self.nx)
        x1 = self.vec_setter(x1, self.nx)
        return np.subtract(x0, x1)

    def _check_if_model_initiated_correctly(self):
        """Checks if all the model parameters are initiated"""
        for key, val in self.phi.items():
            if val is None:
                self.raiseit(f'Parameter -{key}- not assigned!')
        if self.dt is None:
            self.raiseit('Need to assign dt')

# property/setters

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
    def dt(self) -> float:
        """Getter for time interval"""
        return self._dt

    @dt.setter
    def dt(self, in_val: float) -> float:
        """Getter for time interval"""
        self._dt = self.float_setter(in_val)


# Properties

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
