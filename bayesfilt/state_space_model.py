"""Base class for defining motion models"""
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray


class StateSpaceModel(ABC):
    # pylint: disable=invalid-name
    """Base class for defining a state space model"""

    def __init__(
        self,
        nx: int,
        name: str
    ) -> None:
        self.name: str = name  # name of the model
        self._nx: int = self.int_setter(nx)  # dimension of the state
        self._w: ndarray | None = None  # static model parameters
        self._sigmas: ndarray | None = None  # error parameters

    @staticmethod
    def symmetrize(in_mat: ndarray) -> ndarray:
        """Return a symmetrized version of NumPy array"""
        if np.any(np.isnan(in_mat)) or np.any(in_mat.diagonal() < 0.):
            print('\np update went wrong!')
            print(in_mat.diagonal())
        return (in_mat + in_mat.T) / 2.

    def mat_setter(self, in_mat, to_shape=None) -> ndarray:
        """Returns a valid numpy array2d while checking for its shape"""
        in_mat = np.atleast_2d(np.asarray_chkfinite(in_mat, dtype=float))
        if in_mat.ndim != 2:
            self.raiseit(f'Need 2d array, input dim: {in_mat.ndim}')
        if to_shape is not None:
            if in_mat.shape != to_shape:
                print('Shape mismatch!')
                self.raiseit(f'Required: {to_shape}, Input: {in_mat.shape}')
        return in_mat

    def vec_setter(self, in_vec, to_size=None) -> ndarray:
        """Returns a valid numpy array1d while checking for its shape"""
        in_vec = np.atleast_1d(np.asarray_chkfinite(in_vec, dtype=float))
        in_vec = in_vec.flatten()
        if to_size is not None:
            if in_vec.size != to_size:
                print('Size mismatch!')
                self.raiseit(f'Required: {to_size}, Input: {in_vec.size}')
        return in_vec

    def float_setter(self, in_val) -> float:
        """Return a valid scalar"""
        in_val = np.asarray_chkfinite(in_val, dtype=float)
        if in_val.size != 1:
            self.raiseit(f'Need scalar!, change input:{in_val} to scalar')
        return float(in_val.item())

    def int_setter(self, in_val) -> int:
        """Return a valid scalar"""
        in_val = np.asarray_chkfinite(in_val, dtype=int)
        if in_val.size != 1:
            self.raiseit(f'Need scalar!, change input:{in_val} to scalar')
        return int(in_val.item())

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.name}: {outstr}')

    @property
    def w(self) -> ndarray:
        """Getter for model parameter vector w"""
        return self._w

    @property
    def sigmas(self) -> ndarray:
        """Getter for error stds"""
        return self._sigmas

    @property
    def nx(self) -> int:
        """Getter for state dimension"""
        return self._nx
