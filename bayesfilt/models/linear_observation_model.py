"""Classes for defining linear observation models """
# pylint: disable=invalid-name
from typing import List
import numpy as np
from numpy import ndarray
from .observation_model import ObservationModel


class LinearObservationModel(ObservationModel):
    """Class for defining a linear observation model"""

    def __init__(
        self,
        observed_state_inds: List[int],
        **kwargs
    ) -> None:

        # initiate
        self._observed_state_inds = list(set(observed_state_inds))
        super().__init__(
            ny=len(observed_state_inds),
            **kwargs
        )
        if max(self.observed_state_inds) >= self.nx:
            self.raiseit(f'Max state index > {self.nx-1}')
        if len(self.observed_state_inds) > self.nx:
            self.raiseit(f'# of observed states > {nx}')
        self._H: ndarray = np.zeros((self.ny, self.nx))  # obs function
        for k, v in enumerate(self.observed_state_inds):
            self._H[int(k), int(v)] = 1.
        self._J = np.eye(self.ny)

    @property
    def obs_names(self) -> list[str]:
        """Getter for labels"""
        return [self.state_names[i] for i in self.observed_state_inds]

    @property
    def phi_names(self):
        """Declare parameter names of the model"""
        return []

    @property
    def observed_state_inds(self) -> dict:
        """Getter for observation-state pair"""
        return self._observed_state_inds
