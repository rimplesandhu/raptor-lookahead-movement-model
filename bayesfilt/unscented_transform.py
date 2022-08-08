""" Unscented Transform class """
from collections.abc import Callable
import numpy as np
from scipy.linalg import cholesky, sqrtm
from numpy import ndarray


class SigmaPoints:
    # pylint: disable=invalid-name
    """Generates sigma points and weights"""

    def __init__(
        self,
        dim: int,
        alpha: float = 1.,
        beta: float = 0.,
        kappa: float | None = None,
        use_cholesky: bool = False
    ) -> None:
        self._dim: int = dim
        self._alpha: float = alpha
        self._beta: float = beta
        self.use_cholesky = use_cholesky
        self._kappa: float = 3. - self._dim if kappa is None else kappa
        self._lamda: float = alpha**2 * (dim + self._kappa) - dim
        self._wm: ndarray = np.full(2 * dim + 1, 0.5 / (dim + self._lamda))
        self._wc: ndarray = np.full(2 * dim + 1, 0.5 / (dim + self._lamda))
        self._wc[0] = self._lamda / (dim + self._lamda) + (1 - alpha**2 + beta)
        self._wm[0] = self._lamda / (dim + self._lamda)

    def get_sigma_points(
        self,
        m: ndarray | float,
        P: ndarray | float,
        subtract_fn: Callable = np.subtract
    ) -> list[ndarray]:
        """Sigma points"""
        m = np.asarray([m] * self._dim) if np.isscalar(m) else m
        P = np.asarray(np.eye(self._dim) * P) if np.isscalar(P) else P
        m = np.atleast_1d(m)
        P = np.atleast_2d(P)
        if m.shape != (self._dim,):
            raise ValueError(f'm: expected {(self._dim,)}, got {m.shape}')
        if P.shape != (self._dim, self._dim):
            raise ValueError(
                f'P: expected {(self._dim,self._dim)}, got {P.shape}')
        sqrt_method = cholesky if self.use_cholesky else sqrtm
        sqrt_mat = sqrt_method((self._lamda + self._dim) * P)
        #print('sqrt_mat:', np.degrees(sqrt_mat))
        spts = np.zeros((2 * self._dim + 1, self._dim))
        spts[0] = m
        for k in range(self._dim):
            spts[k + 1] = subtract_fn(m, sqrt_mat[k])
            spts[self._dim + k + 1] = subtract_fn(m, -sqrt_mat[k])
        return list(spts)

    def __repr__(self) -> str:
        """Print output"""
        out_str = ':::SigmaPoints\n'
        out_str += f'dimension = {self._dim}\n'
        out_str += f'alpha = {self._alpha}\n'
        out_str += f'beta = {self._beta}\n'
        out_str += f'kappa = {self._kappa}\n'
        out_str += f'lambda = {self._lamda}\n'
        out_str += f'weights (m) = {self._wm}\n'
        out_str += f'weights (c) = {self._wc}'
        return out_str

    @ property
    def wm(self) -> ndarray:
        """Weights for mean"""
        return self._wm

    @ property
    def wc(self) -> ndarray:
        """Weights for cov"""
        return self._wc

    @ property
    def alpha(self) -> float:
        """Alpha parameter"""
        return self._alpha

    @ property
    def kappa(self) -> float:
        """Beta parameter"""
        return self._kappa

    @ property
    def lamda(self) -> float:
        """Lambda parameter"""
        return self._lamda

    @ property
    def beta(self) -> float:
        """Beta parameter"""
        return self._beta


class UnscentedTransform(SigmaPoints):
    # pylint: disable=invalid-name
    """Unscented Transform"""

    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

    def transform(
        self,
        m: ndarray,
        P: ndarray,
        nl_func: Callable,
        x_subtract: Callable = np.subtract,
        y_subtract: Callable = np.subtract,
        mean_fn: Callable | None = None
    ) -> tuple[ndarray, ndarray]:
        """Mean and covariance resulting from the unscented Transform"""
        x_spts = self.get_sigma_points(m, P, x_subtract)
        #print('x:', np.degrees(x_spts))
        x_res = [x_subtract(ix, m) for ix in x_spts]
        #print('res:', np.degrees(x_res))
        y_spts = [np.atleast_1d(nl_func(ix)) for ix in x_spts]
        # mean function should handle angle
        if mean_fn is None:
            # y_mvec = np.dot(self.wm, y_spts)
            y_mvec = sum([iw * iy for iw, iy in zip(self.wm, y_spts)])
        else:
            y_mvec = mean_fn(y_spts, self.wm)
            #print('y:', np.degrees(y_spts))
        y_res = [y_subtract(iy, y_mvec) for iy in y_spts]
        Pyy = sum([iw * np.outer(iy, iy) for iy, iw in zip(y_res, self.wc)])
        # if np.any(np.linalg.eigvals(Pyy) < 0):
        #     print('P gone wrong in ut!')
        #print('m:', np.around(m, 2), '\nP:', np.around(P.diagonal(), 2))
        # print('mnew:', np.around(y_mvec, 2),
        #      '\nPnew:', np.around(Pyy.diagonal(), 2))
        # for i, ipt in enumerate(x_spts):
        #     print(i, 'xpt:', np.around(ipt, 2))
        #     print(i, 'ypt:', np.around(y_spts[i], 2))
        Pxy = sum([iw * np.outer(ix, iy)
                   for ix, iy, iw in zip(x_res, y_res, self.wc)])
        return y_mvec, Pyy, Pxy

    # def compute_output_mean_and_cov(
    #     self,
    #     spts: ndarray
    # ) -> tuple[ndarray, ndarray]:
    #     """Compute mean and covariance of the output"""
    #     spts = np.atleast_2d(spts)
    #     spts = spts.T if spts.shape[1] == 2 * self._dim + 1 else spts
    #     mvec = np.dot(self._wm, spts)
    #     residual = spts - mvec[np.newaxis, :]
    #     Pmat = np.dot(residual.T, np.dot(np.diag(self._wc), residual))
    #     return mvec, Pmat
