"""Base class for defining state space models"""
# pylint: disable=invalid-name
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
from numpy import ndarray


class StateSpaceModel(ABC):
    """Base class for defining a state space model"""

    def __init__(
        self,
        nx: int,
        name: str = 'M0',
        verbose: bool = False
    ) -> None:
        self.name: str = name  # name of the model
        self._nx: int = self.scaler(nx, dtype='int32')  # dimension of state
        self._state_names = [f'x_{i}' for i in range(self.nx)]
        self._phi = ParameterDict(self.phi_names)  # parameter dict
        self.verbose = verbose

    def __repr__(self):
        out_str = f'----{self.name}----\n'
        out_str += f'States    ({self.nx}): ' + \
            ','.join(self.state_names) + '\n'
        out_str += f'Parameters({len(self.phi_names)}): '
        out_str += ', '.join([f'{k}={v}' for k, v in self.phi.items()]) + '\n'
        return out_str

    @property
    @abstractmethod
    def phi_names(self):
        """Child class must define the names of static parameters"""

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
                self.raiseit(f'Parameter -{k}- not found in {self.phi_names}!')
            else:
                self._phi[k] = v

    @property
    def state_names(self) -> List[str]:
        """Getter for state names"""
        return self._state_names

    @state_names.setter
    def state_names(self, in_list) -> None:
        """Setter for state names"""
        if len(in_list) != self.nx:
            self.raiseit(f'Number of state labels should be {self.nx}')
        self._state_names = in_list

    @property
    def nx(self) -> int:
        """Getter for state dimension"""
        return self._nx

    def check_ready_to_deploy(self) -> bool:
        """Checks if all the model parameters/settings are initiated"""
        for key, val in self.phi.items():
            if val is None:
                self.raiseit(f'Parameter -{key}- not assigned!')

    def symmetrize(self, in_mat: ndarray) -> ndarray:
        """Return a symmetrized version of NumPy array"""
        if np.any(np.isnan(in_mat)):
            self.raiseit('\nRogue matrix: nan entries found!')
        return (in_mat + in_mat.T) / 2.

    def matrix(
        self,
        in_mat: ndarray,
        shape: List[int] | None = None,
        dtype: str = 'float64'
    ) -> ndarray:
        """Returns a valid numpy array while checking for its shape"""
        in_mat = np.atleast_1d(np.asarray_chkfinite(in_mat, dtype=dtype))
        # shape = shape if isinstance(shape, list) else (shape,)
        # if in_mat.ndim != len(shape):
        #     self.raiseit(f'Dim {in_mat.ndim} not equal to {len(shape)}')
        if shape is not None:
            if in_mat.shape != shape:
                self.raiseit(f'Required: {shape}, Input: {in_mat.shape}')
        return in_mat

    def scaler(self, in_val, dtype: str = 'float64') -> float | int:
        """Return a valid scalar"""
        in_val = np.asarray_chkfinite(in_val, dtype=dtype)
        if in_val.size != 1:
            self.raiseit(f'Need scalar!, change {in_val} to scalar')
        return in_val.item()

    def valid_list(self, in_list, length) -> List:
        """Return a valid list"""
        assert isinstance(in_list, list), self.raiseit('Need a list')
        if len(in_list) != length:
            self.raiseit(f'Need list {in_list} of length {length}!')
        return in_list

    def raiseit(self, outstr: str = "", exception=ValueError) -> None:
        """Raise exception with the out string"""
        raise exception(f'{self.__class__.__name__}: {outstr}')

    # def warnit(self, outstr: str = "") -> None:
    #     """Raise warning with the out string"""
    #     warnings.warn(f'{self.name}: {outstr}')


class ParameterDict(dict):
    """Class for defining parameter dictionary"""

    def __init__(self, list_of_names):
        super().__init__({k: None for k in list_of_names})

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        # istr = f'Parameter {key} not found! Choose from {list(self.keys())}'
        # assert key in self.keys(), istr
        dict.__setitem__(self, key, val)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __repr__(self):
        return dict.__repr__(self)
