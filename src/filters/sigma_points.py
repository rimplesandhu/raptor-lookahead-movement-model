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
    kappa: float
    use_cholesky: bool
    fun_subtract_in: Callable = field(default=np.subtract, repr=False)

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
            self.raiseit(f'P: expected {(self.dim, self.dim)}, got {P.shape}')
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
