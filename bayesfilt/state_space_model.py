"""Base class for defining state space models"""
from abc import ABC, abstractmethod
from typing import Dict, List
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
        self._state_names = [f'x_{i}' for i in range(self.nx)]
        self._phi = {k: None for k, _ in self.phi_definition.items()}

    @property
    @abstractmethod
    def phi_definition(self):
        """Declare parameter names of the model"""

    def _print_info(self):
        out_str = f'----{self.name}----\n'
        out_str += 'State: ' + ', '.join(self.state_names) + '\n'
        out_str += 'Parameters: '
        out_str += ', '.join([f'{k}={v}' for k, v in self.phi.items()]) + '\n'
        return out_str

    @property
    def phi(self) -> Dict[str, float]:
        """Getter for model parameter vector w"""
        return self._phi

    @phi.setter
    def phi(self, iz) -> None:
        """Getter for model parameter vector w"""
        for k, v in self.phi_definition.items():
            assert k in iz, self.raiseit(f'Cant find {k} in {iz}')
            istr = f'Incorrect size for parameter {k}, required size {v}, '
            istr += f'input size {np.array(iz[k]).size}'
            assert int(v) == np.array(iz[k]).size, self.raiseit(istr)
        self._phi = {k: v for k, v in iz.items() if k in self.phi_definition}

    @ property
    def nx(self) -> int:
        """Getter for state dimension"""
        return self._nx

    @ property
    def state_names(self) -> List[str]:
        """Getter for state names"""
        return self._state_names

    @ state_names.setter
    def state_names(self, in_list) -> None:
        """Setter for state names"""
        if len(in_list) != self.nx:
            self.raiseit(f'Number of state labels should be {self.nx}')
        self._state_names = in_list

    @ staticmethod
    def symmetrize(in_mat: ndarray) -> ndarray:
        """Return a symmetrized version of NumPy array"""
        if np.any(np.isnan(in_mat)) or np.any(in_mat.diagonal() < 0.):
            print('\np update went wrong!')
            print(in_mat.diagonal())
        return (in_mat + in_mat.T) / 2.

    @ staticmethod
    def check_symmetric(a: ndarray, rtol=1e-05, atol=1e-08):
        """check if matrix is symmetric or not"""
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

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
