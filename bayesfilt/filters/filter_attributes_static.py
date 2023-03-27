""" Base class for defining Bayesian filtering attributes """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from typing import Callable
from functools import partial
from dataclasses import dataclass, field
import numpy as np
from numpy import ndarray
from .utils import validate_array
TypeFunc21 = Callable[(ndarray, ndarray), ndarray]


@dataclass(frozen=True)
class FilterAttributesStatic:
    """Static attributes of a filter"""
    nx: int
    ny: int
    dt: float
    dt_tol: float | None = None
    epsilon: float = 1e-6
    state_names: list[str] | None = None
    verbose: bool = False
    pars: dict[str, float] = field(default_factory=dict, repr=True)

    # model functions
    fun_f: TypeFunc21 | None = field(default=None, repr=False)
    fun_Fjac: TypeFunc21 | None = field(default=None, repr=False)
    fun_Gjac: TypeFunc21 | None = field(default=None, repr=False)
    fun_h: TypeFunc21 | None = field(default=None, repr=False)
    fun_Hjac: TypeFunc21 | None = field(default=None, repr=False)
    fun_Jjac: TypeFunc21 | None = field(default=None, repr=False)
    fun_Q: TypeFunc21 | None = field(default=None, repr=False)

    # matrices
    mat_F: ndarray | None = field(default=None, repr=False)
    mat_G: ndarray | None = field(default=None, repr=False)
    mat_H: ndarray | None = field(default=None, repr=False)
    mat_J: ndarray | None = field(default=None, repr=False)
    mat_Q: ndarray | None = field(default=None, repr=False)
    vec_qbar: ndarray | None = field(default=None, repr=False)
    vec_rbar: ndarray | None = field(default=None, repr=False)

    # add/subtract functions
    fun_subtract_x: TypeFunc21 = field(default=np.subtract, repr=False)
    fun_subtract_y: TypeFunc21 = field(default=np.subtract, repr=False)
    fun_weighted_mean_x: TypeFunc21 = field(
        default=partial(np.average, axis=0), repr=False)
    fun_weighted_mean_y: TypeFunc21 = field(
        default=partial(np.average, axis=0), repr=False)

    def __post_init__(self):
        """post initiation function"""

        # default values
        if self.dt_tol is None:
            object.__setattr__(self, 'dt_tol', self.dt / 2.)
        if self.vec_qbar is None:
            object.__setattr__(self, 'vec_qbar', np.zeros(self.nx))
        if self.vec_rbar is None:
            object.__setattr__(self, 'vec_rbar', np.zeros(self.ny))
        if self.state_names is None:
            object.__setattr__(self, 'state_names',
                               [f'x_{i}' for i in range(self.nx)])

        # mat_F takes priority over everything
        if self.mat_G is None:
            object.__setattr__(self, 'mat_G', np.eye(self.nx))
            if self.fun_Gjac is None:
                object.__setattr__(
                    self, 'fun_Gjac', partial(self.v2m, self.mat_G))

        if self.mat_J is None:
            object.__setattr__(self, 'mat_J', np.eye(self.ny))
            if self.fun_Jjac is None:
                object.__setattr__(
                    self, 'fun_Jjac', partial(self.v2m, self.mat_J))

        if (self.mat_F is not None) and (self.fun_f is None):
            object.__setattr__(self, 'fun_f', partial(self.v2v, self.mat_F))
            object.__setattr__(self, 'fun_Fjac', partial(self.v2m, self.mat_F))

        # mat_H takes priority over everything
        if (self.mat_H is not None) and (self.fun_h is None):
            object.__setattr__(self, 'fun_h', partial(self.v2v, self.mat_H))
            object.__setattr__(self, 'fun_Hjac', partial(self.v2m, self.mat_H))

        if self.mat_Q is not None:
            object.__setattr__(self, 'fun_Q', partial(self.v2m, self.mat_Q))

        self.check_valid_initialization()
        self.check_valid_matrices()

    def check_valid_matrices(self):
        """Check validity of matrices"""
        if self.mat_F is not None:
            validate_array(self.mat_F, (self.nx, self.nx))
        if self.mat_G is not None:
            validate_array(self.mat_G, (self.nx, self.nx))
        if self.mat_Q is not None:
            validate_array(self.mat_Q, (self.nx, self.nx))
        if self.mat_H is not None:
            validate_array(self.mat_H, (self.ny, self.nx))
        if self.mat_J is not None:
            validate_array(self.mat_J, (self.ny, self.ny))
        validate_array(self.vec_rbar, (self.ny,))
        validate_array(self.vec_qbar, (self.nx,))

    def check_valid_initialization(self):
        """Check if filter is properly initialized"""
        if (self.mat_F is None) and (self.fun_f is None):
            self.raiseit('Either mat_F or fun_f required to initiate filter!')
        if (self.mat_Q is None) and (self.fun_Q is None):
            self.raiseit('Either mat_Q or fun_Q required to initiate filter!')
        if (self.mat_H is None) and (self.fun_h is None):
            self.raiseit('Either mat_H or fun_h required to initiate filter!')

    def v2m(
        self,
        mat: ndarray | None = None,
        x: ndarray | None = None,
        u: ndarray | None = None,
    ):
        """dummy funcion that takes in state vector and return a matrix"""
        return mat

    def v2v(
        self,
        mat: ndarray | None = None,
        x: ndarray | None = None,
        u: ndarray | None = None,
    ):
        """dummpy funcion that takes in state vector and return a vector"""
        return mat @ x

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.__class__.__name__}: {outstr}')
