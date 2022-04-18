""" Classes for defining linear motion models """
from abc import abstractmethod
import numpy as np
from numpy import ndarray
from typing import Sequence


class LinearObservationModel:
    """ Abstract class for representing an observation model"""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        obs_states: Sequence[int]
    ) -> None:
        self._state_dim: int = state_dim
        self._obs_dim: int = obs_dim
        self._R: ndarray = np.eye(obs_dim)  # obs error covariance matrix
        self._H: ndarray = np.zeros((obs_dim, state_dim))
        obs_states = np.atleast_1d(obs_states)
        assert obs_states.size <= state_dim, 'Size(mstates)>state_dim!'
        assert obs_states.size == obs_dim, 'Size(mstates) != obs_dim!'
        for i, j in enumerate(np.atleast_1d(obs_states)):
            self._H[i, j] = 1.

    @abstractmethod
    def compute(self, obs_error_stds: ndarray) -> None:
        """ Computes updated model matrices """
        pass

    @property
    def state_dim(self) -> int:
        """Returns dimension (size) of the state vector"""
        return self._state_dim

    def obs_dim(self) -> int:
        """Returns dimension (size) of the observation vector"""
        return self._obs_dim

    @property
    def H(self) -> ndarray:
        """Returns observation matrix F"""
        return self._H

    @property
    def R(self) -> ndarray:
        """Returns observation error covariance matrix Q"""
        return self._R

    def __str__(self):
        out_str = ':::Linear observation model\n'
        out_str += f'State dimension: {self._state_dim}\n'
        out_str += f'Obs dimension: {self._obs_dim}\n'
        out_str += f'Obs mat H:\n {np.array_str(self._H, precision=3)}\n'
        out_str += f'Error cov mat R:\n {np.array_str(self._R, precision=3)}\n'
        return out_str
