""" Unscented Transform class """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from dataclasses import dataclass, field
from typing import Callable
from functools import partial
import numpy as np
from numpy import ndarray
from .sigma_points import SigmaPoints


@dataclass(frozen=True)
class UnscentedTransform(SigmaPoints):
    """Unscented Transform"""
    model_fun: Callable | None = None
    fun_subtract_out: Callable = field(default=np.subtract, repr=False)
    fun_mean_out: Callable = field(
        default=partial(np.average, axis=0), repr=False)

    def __post_init__(self):
        """post initiation function"""
        super().__post_init__()
        if self.model_fun is None:
            self.raiseit('model_fun required to initiate unscented transform!')

    def transform(
        self,
        m: ndarray,
        P: ndarray
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Mean and covariance resulting from the unscented Transform"""
        x_spts = self.get_sigma_points(m, P)
        # print([np.degrees(ix) for ix in x_spts])
        y_spts = [np.atleast_1d(self.model_fun(ix)) for ix in x_spts]
        # print([np.degrees(ix) for ix in y_spts])
        y_mvec = self.fun_mean_out(y_spts, weights=self.wm)
        # print(np.degrees(y_mvec))
        x_res = [self.fun_subtract_in(ix, m) for ix in x_spts]
        y_res = [self.fun_subtract_out(iy, y_mvec) for iy in y_spts]
        Pyy = sum([iw * np.outer(iy, iy) for iy, iw in zip(y_res, self.wc)])
        Pxy = sum([iw * np.outer(ix, iy)
                   for ix, iy, iw in zip(x_res, y_res, self.wc)])
        return y_mvec, Pyy, Pxy


# class UnscentedTransform(SigmaPoints):
#     """Unscented Transform"""

#     def transform(
#         self,
#         m: ndarray,
#         P: ndarray,
#         model_fun: Callable,
#         self.fun_subtract: Callable = np.subtract,
#         fun_subtract_out: Callable = np.subtract,
#         fun_mean_out: Callable | None = None
#     ) -> tuple[ndarray, ndarray]:
#         """Mean and covariance resulting from the unscented Transform"""
#         x_spts = self.get_sigma_points(m, P, self.fun_subtract)
#         x_res = [self.fun_subtract(ix, m) for ix in x_spts]
#         y_spts = [np.atleast_1d(model_fun(ix)) for ix in x_spts]
#         if fun_mean_out is None:
#             # y_mvec = np.dot(self.wm, y_spts)
#             y_mvec = sum([iw * iy for iw, iy in zip(self.wm, y_spts)])
#         else:
#             y_mvec = fun_mean_out(y_spts, self.wm)
#             # print('y:', np.degrees(y_spts))
#         y_res = [fun_subtract_out(iy, y_mvec) for iy in y_spts]
#         Pyy = sum([iw * np.outer(iy, iy) for iy, iw in zip(y_res, self.wc)])
#         # if np.any(np.linalg.eigvals(Pyy) < 0):
#         #     print('P gone wrong in ut!')
#         # print('m:', np.around(m, 2), '\nP:', np.around(Pyy.diagonal(), 2))
#         # print(Pyy)
#         # print('mnew:', np.around(y_mvec, 2),
#         #      '\nPnew:', np.around(Pyy.diagonal(), 2))
#         # for i, ipt in enumerate(x_spts):
#         #     print(i, 'xpt:', np.around(ipt, 2))
#         #     print(i, 'ypt:', np.around(y_spts[i], 2))
#         Pxy = sum([iw * np.outer(ix, iy)
#                    for ix, iy, iw in zip(x_res, y_res, self.wc)])
#         return y_mvec, Pyy, Pxy
