"""Base class for defining state space models"""
# pylint: disable=invalid-name
from abc import ABC, abstractmethod, abstractproperty
from typing import Dict, List
import warnings
import numpy as np
from numpy import ndarray


class ParameterDict(dict):
    """Class for defining parameter dictionary"""

    def __init__(self, list_of_names):
        super().__init__({k: None for k in list_of_names})

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        #istr = f'Parameter {key} not found! Choose from {list(self.keys())}'
        #assert key in self.keys(), istr
        dict.__setitem__(self, key, val)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __repr__(self):
        return dict.__repr__(self)


class StateSpaceModel(ABC):
    """Base class for defining a state space model"""

    def __init__(
        self,
        nx: int,
        name: str = 'StateSpaceModel'
    ) -> None:
        self.name: str = name  # name of the model
        self._nx: int = self.int_setter(nx)  # dimension of the state
        self._state_names = [f'x_{i}' for i in range(self.nx)]
        self._phi = ParameterDict(self.phi_names)  # parameter dict

    def __str__(self):
        out_str = f'----{self.name}----\n'
        out_str += f'State({self.nx}): ' + ', '.join(self.state_names) + '\n'
        out_str += f'Parameters({len(self.phi_names)}): '
        out_str += ', '.join([f'{k}={v}' for k, v in self.phi.items()]) + '\n'
        return out_str

# abstact methods
    @property
    @abstractmethod
    def phi_names(self):
        """Child class must define the names of static parameters"""

# getter/setters
    @property
    def phi(self) -> Dict[str, float]:
        """Getter for model parameter vector w"""
        return self._phi

    @phi.setter
    def phi(self, in_dict) -> None:
        """Getter for model parameter vector w"""
        if not isinstance(in_dict, dict):
            self.raiseit(f'Input {in_dict} should be a dict!')
        for k, v in in_dict.items():
            if k not in self.phi_names:
                self.warnit(f'Parameter -{k}- not found in {self.phi_names}!')
            else:
                self._phi[k] = v

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

# property

    @ property
    def nx(self) -> int:
        """Getter for state dimension"""
        return self._nx

# static methods

    @staticmethod
    def symmetrize(in_mat: ndarray) -> ndarray:
        """Return a symmetrized version of NumPy array"""
        if np.any(np.isnan(in_mat)) or np.any(in_mat.diagonal() < 0.):
            print('\np update went wrong!')
            print(in_mat.diagonal())
        return (in_mat + in_mat.T) / 2.

    @staticmethod
    def check_symmetric(a: ndarray, rtol=1e-05, atol=1e-08):
        """check if matrix is symmetric or not"""
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

# functions

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

    def warnit(self, outstr: str = "") -> None:
        """Raise warning with the out string"""
        warnings.warn(f'{self.name}: {outstr}')
