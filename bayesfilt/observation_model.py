"""Classes for defining an observation model"""
from abc import abstractmethod
import numpy as np
from numpy import ndarray
from .state_space_model import StateSpaceModel


class ObservationModel(StateSpaceModel):
    # pylint: disable=invalid-name
    """Class for defining an observation model"""

    def __init__(
        self,
        nx: int,
        ny: int,
        name: str = 'obs_model',
        observed: dict | None = None
    ) -> None:

        # model parameters
        super().__init__(nx=nx, name=name)
        self._ny = self.int_setter(ny)  # dim of obs vector
        self._observed = observed  # obs-state indices pair
        self._labels = [f'y_{i}' for i in range(self.nx)]

        # model matrices
        self._H: ndarray | None = None  # Observation-State matrix
        self._J: ndarray | None = None  # Error Jacobian matrix
        self._R: ndarray | None = None  # Error covariance matrix

        if self.observed is not None:
            if max(self.observed.keys()) >= self.nx:
                self.raiseit(f'Max state index cannot exceed {self.nx-1}')
            if max(self.observed.values()) >= self.ny:
                self.raiseit(f'Max obs index cannot exceed {self.ny-1}')

    @abstractmethod
    def update(
        self,
        sigmas: ndarray,
        R: ndarray,
        w: ndarray
    ) -> None:
        """Update system parameters"""

    @abstractmethod
    def h(
        self,
        x: ndarray | None,
        r: ndarray | None
    ) -> ndarray:
        """Measurement equation"""

    @abstractmethod
    def compute_H(
        self,
        x: ndarray | None,
        r: ndarray | None
    ) -> ndarray:
        """Get H matrix"""

    @abstractmethod
    def compute_J(
        self,
        x: ndarray | None,
        r: ndarray | None
    ) -> ndarray:
        """Get J matrix"""

    @property
    def ny(self) -> int:
        """Dimension of observation space """
        return self._ny

    @property
    def observed(self) -> dict:
        """Getter for observation-state pair"""
        return self._observed

    @property
    def H(self) -> ndarray:
        """Measurement-State matrix"""
        return self._H

    @property
    def J(self) -> ndarray:
        """Measurement matrix"""
        return self._J

    @property
    def R(self) -> ndarray:
        """Measurement error covariance matrix"""
        return self._R

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

    def __str__(self):
        out_str = f':::{self.name}\n'
        out_str += f'State Dimension: {self._nx}\n'
        out_str += f'Observation Dimension: {self._ny}\n'
        out_str += f'H:\n {np.array_str(np.array(self._H), precision=3)}\n'
        out_str += f'J:\n {np.array_str(np.array(self._J), precision=3)}\n'
        out_str += f'R:\n {np.array_str(np.array(self._R), precision=3)}\n'
        return out_str
