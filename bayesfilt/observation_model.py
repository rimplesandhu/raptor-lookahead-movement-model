"""Classes for defining an observation model"""
from collections.abc import Callable
from numpy import ndarray
import numpy as np
from .state_space_model import StateSpaceModel


class ObservationModel(StateSpaceModel):
    # pylint: disable=invalid-name
    """Class for defining an observation model"""

    def __init__(
        self,
        nx: int,
        ny: int,
        name: str = 'ObservationModel'
    ) -> None:

        # model parameters
        super().__init__(nx=nx, name=name)
        self._ny = self.int_setter(ny)  # dimension of observation vector
        self._obs_names = [f'y_{i}' for i in range(self.ny)]

        # model matrices
        self._H: ndarray | Callable | None = None  # Observation-State matrix
        self._J: ndarray | Callable | None = None  # Error Jacobian matrix
        self._R: ndarray | Callable | None = None  # Error covariance matrix

    def __str__(self):
        out_str = StateSpaceModel.__str__(self)
        out_str += f'Observations({self.ny}): ' + \
            ', '.join(self.obs_names) + '\n'
        out_str += f'H:\n {np.array_str(np.array(self.H), precision=3)}\n'
        out_str += f'J:\n {np.array_str(np.array(self.J), precision=3)}\n'
        out_str += f'R:\n {np.array_str(np.array(self.R), precision=3)}\n'
        return out_str

    @property
    def ny(self) -> int:
        """Dimension of observation space """
        return self._ny

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

    @R.setter
    def R(self, in_mat: ndarray) -> ndarray:
        """Measurement error covariance matrix"""
        self._R = self.mat_setter(in_mat, (self.ny, self.ny))
        assert self.check_symmetric(self.R), f'{self.name}: R not symmetric!'
        # if not self.check_symmetric(self.R):
        #     print(f'{self.name}: R matrix not symmetric!')

    @property
    def obs_names(self) -> list[str]:
        """Getter for labels"""
        return self._obs_names

    @obs_names.setter
    def obs_names(self, in_list) -> None:
        """Setter for labels"""
        if len(in_list) != self.ny:
            self.raiseit(f'Number of observation labels should be {self.ny}')
        self._obs_names = in_list
