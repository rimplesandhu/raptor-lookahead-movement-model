"""Classes for defining linear motion/observation models """
# pylint: disable=invalid-name
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from ._base_model import StateSpaceModel


class LinearObservationModel(StateSpaceModel):
    """Class for linear observation motion model with additive Gaussain errors"""

    def __init__(
            self,
            nx: int,
            obs_state_inds: list[int],
            xnames: list[str] = None
    ):
        """Initiatilization function"""
        self._state_inds = np.atleast_1d(obs_state_inds)
        _names = [f'X{i}' for i in range(nx)]
        super().__init__(
            nx=int(nx),
            name=f'LinearObsModel({nx}/{len(self._state_inds)})',
            xnames=_names if xnames is None else xnames
        )
        if max(self._state_inds) >= self.nx:
            self.raiseit(f'Max state index > {self.nx-1}')
        if len(self._state_inds) > self.nx:
            self.raiseit(f'# of observed states > {self.nx}')

        # H matrix
        _Hmat: ndarray = np.zeros((self.ny, self.nx))  # obs function
        for k, v in enumerate(self.obs_state_inds):
            _Hmat[int(k), int(v)] = 1.

    @property
    def ny(self) -> int:
        """Dimension of observation space """
        return len(self.obs_state_inds)

    @property
    def obs_state_inds(self) -> int:
        """Observed stae indices"""
        return self._state_inds

    @property
    def Hmat(self) -> ndarray:
        """Measurement-State matrix"""
        _Hmat: ndarray = np.zeros((self.ny, self.nx))  # obs function
        for k, v in enumerate(self.obs_state_inds):
            _Hmat[int(k), int(v)] = 1.
        return _Hmat

    @property
    def ynames(self) -> list[str]:
        """Observed state names"""
        return [self.xnames[i] for i in self.obs_state_inds]
