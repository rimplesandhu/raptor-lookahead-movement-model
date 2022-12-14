"""Classes for defining linear observation models """
from abc import abstractmethod
from typing import List
import numpy as np
from numpy import ndarray
from .observation_model import ObservationModel


class LinearObservationModel(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
        nx: int,
        observed_state_inds: List[int],
        name: str = 'LinearObservationModel'
    ) -> None:

        # initiate
        self._observed_state_inds = list(set(observed_state_inds))
        super().__init__(
            nx=nx,
            ny=len(observed_state_inds),
            name=name
        )
        if max(self.observed_state_inds) >= nx:
            self.raiseit(f'Max state index > {nx-1}')
        if len(self.observed_state_inds) > nx:
            self.raiseit(f'# of observed states > {nx}')
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function
        for k, v in enumerate(self.observed_state_inds):
            self._H[int(k), int(v)] = 1.
        self._J = np.eye(self.ny)

    @property
    def phi_definition(self):
        """Declare parameter names of the model"""
        return {}

    @property
    def observed_state_inds(self) -> dict:
        """Getter for observation-state pair"""
        return self._observed_state_inds

    def __str__(self):
        out_str = super()._print_info()
        out_str += f'Observated states: {self.observed_state_inds}\n'
        out_str += f'H:\n {np.array_str(np.array(self._H), precision=4)}\n'
        out_str += f'J:\n {np.array_str(np.array(self._J), precision=4)}\n'
        out_str += f'R:\n {np.array_str(np.array(self._R), precision=4)}\n'
        return out_str
