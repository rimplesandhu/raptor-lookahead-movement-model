""" Sigma points class """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from scipy.linalg import cholesky, sqrtm
from numpy import ndarray


@dataclass(frozen=True)
class SigmaPoints:
    """Sigma Points"""
    dim: int
    alpha: float
    beta: float
    kappa: float | None = None
    use_cholesky: bool = field(default=False, repr=False)
    fun_subtract_in: Callable = field(default=np.subtract, repr=False)

    def __post_init__(self):
        """post initiation function"""
        if self.kappa is None:
            object.__setattr__(self, 'kappa', 3. - self.dim)

    @property
    def lamda(self) -> float:
        """Lambda"""
        return self.alpha**2 * (self.dim + self.kappa) - self.dim

    @property
    def wm(self):
        """weights for mean"""
        wm_vec = np.full(2 * self.dim + 1, 0.5 / (self.dim + self.lamda))
        wm_vec[0] = self.lamda / (self.dim + self.lamda)
        return wm_vec

    @property
    def wc(self):
        """weights for cov"""
        wc_vec = np.full(2 * self.dim + 1, 0.5 / (self.dim + self.lamda))
        wc_vec[0] = self.lamda / (self.dim + self.lamda)
        wc_vec[0] += (1 - self.alpha**2 + self.beta)
        return wc_vec

    def get_sigma_points(
        self,
        m: ndarray | float,
        P: ndarray | float,
    ) -> list[ndarray]:
        """Sigma points"""
        m = np.asarray([m] * self.dim) if np.isscalar(m) else m
        P = np.eye(self.dim) * P if np.isscalar(P) else P
        m = np.atleast_1d(m)
        P = np.atleast_2d(P)
        if m.shape != (self.dim,):
            self.raiseit(f'mean vec: expected {(self.dim,)}, got {m.shape}')
        if P.shape != (self.dim, self.dim):
            self.raiseit(f'P: expected {(self.dim,self.dim)}, got {P.shape}')
        sqrt_method = cholesky if self.use_cholesky else sqrtm
        sqrt_mat = sqrt_method((self.lamda + self.dim) * P)
        spts = np.zeros((2 * self.dim + 1, self.dim))
        spts[0] = m
        for k in range(self.dim):
            spts[k + 1] = self.fun_subtract_in(m, sqrt_mat[k])
            spts[self.dim + k + 1] = self.fun_subtract_in(m, -sqrt_mat[k])
        return list(spts)

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.__class__.__name__}: {outstr}')


# class SigmaPointsOld:
#     """Sigma points class"""

#     def __init__(
#         self,
#         dim: int,
#         alpha: float,
#         beta: float,
#         kappa: float | None = None,
#         use_cholesky: bool = False
#     ) -> None:
#         self._dim: int = dim
#         self._alpha: float = alpha
#         self._beta: float = beta
#         self.use_cholesky = use_cholesky
#         self._kappa: float = 3. - self._dim if kappa is None else kappa
#         self._lamda: float = alpha**2 * (dim + self._kappa) - dim
#         self._wm: ndarray = np.full(2 * dim + 1, 0.5 / (dim + self._lamda))
#         self._wc: ndarray = np.full(2 * dim + 1, 0.5 / (dim + self._lamda))
#         self._wc[0] = self._lamda / (dim + self._lamda) + (1 - alpha**2 + beta)
#         self._wm[0] = self._lamda / (dim + self._lamda)

#     def get_sigma_points(
#         self,
#         m: ndarray | float,
#         P: ndarray | float,

#     ) -> list[ndarray]:
#         """Sigma points"""
#         m = np.asarray([m] * self._dim) if np.isscalar(m) else m
#         P = np.asarray(np.eye(self._dim) * P) if np.isscalar(P) else P
#         m = np.atleast_1d(m)
#         P = np.atleast_2d(P)
#         if m.shape != (self._dim,):
#             raise ValueError(f'm: expected {(self._dim,)}, got {m.shape}')
#         if P.shape != (self._dim, self._dim):
#             raise ValueError(
#                 f'P: expected {(self._dim,self._dim)}, got {P.shape}')
#         sqrt_method = cholesky if self.use_cholesky else sqrtm
#         sqrt_mat = sqrt_method((self._lamda + self._dim) * P)
#         # print('sqrt_mat:', np.degrees(sqrt_mat))
#         spts = np.zeros((2 * self._dim + 1, self._dim))
#         spts[0] = m
#         for k in range(self._dim):
#             spts[k + 1] = subtract_fn(m, sqrt_mat[k])
#             spts[self._dim + k + 1] = subtract_fn(m, -sqrt_mat[k])
#         return list(spts)

#     def __repr__(self) -> str:
#         """Print output"""
#         out_str = ':::SigmaPoints\n'
#         out_str += f'dimension = {self._dim}\n'
#         out_str += f'alpha = {self._alpha}\n'
#         out_str += f'beta = {self._beta}\n'
#         out_str += f'kappa = {self._kappa}\n'
#         out_str += f'lambda = {self._lamda}\n'
#         out_str += f'weights (m) = {self._wm}\n'
#         out_str += f'weights (c) = {self._wc}'
#         return out_str

#     @ property
#     def wm(self) -> ndarray:
#         """Weights for mean"""
#         return self._wm

#     @ property
#     def wc(self) -> ndarray:
#         """Weights for cov"""
#         return self._wc

#     @ property
#     def alpha(self) -> float:
#         """Alpha parameter"""
#         return self._alpha

#     @ property
#     def kappa(self) -> float:
#         """Beta parameter"""
#         return self._kappa

#     @ property
#     def lamda(self) -> float:
#         """Lambda parameter"""
#         return self._lamda

#     @ property
#     def beta(self) -> float:
#         """Beta parameter"""
#         return self._beta
