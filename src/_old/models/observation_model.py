"""Base class for defining an observation model"""
# pylint: disable=invalid-name
from typing import Callable, List
from numpy import ndarray
import numpy as np
from .state_space_model import StateSpaceModel


class ObservationModel(StateSpaceModel):
    """Class for defining an observation model"""

    def __init__(
        self,
        nx: int,
        ny: int,
        ignore_inds_for_loglik: List[int] | None = None,
        name: str = 'ObservationModel',
        verbose: bool = False
    ) -> None:

        # model parameters
        super().__init__(nx=nx, name=name, verbose=verbose)
        self._ny = self.scaler(ny, dtype='int32')
        self._obs_names = [f'y_{i}' for i in range(self.ny)]
        self.ignore_inds_for_loglik = ignore_inds_for_loglik

        # model matrices
        self._H: ndarray | Callable | None = None  # Observation-State matrix
        self._J: ndarray | Callable | None = None  # Error Jacobian matrix
        self._R: ndarray | Callable | None = None  # Error covariance matrix

    def __repr__(self):
        out_str = super().__repr__()
        out_str += f'Obs States({self.ny}): ' + ','.join(self.obs_names) + '\n'
        if self.verbose:
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
        self._R = self.matrix(in_mat, (self.ny, self.ny), dtype='float64')

    @property
    def obs_names(self) -> list[str]:
        """Getter for labels"""
        return self._obs_names

    @obs_names.setter
    def obs_names(self, in_list) -> None:
        """Setter for labels"""
        self._obs_names = self.valid_list(in_list, self.ny)
