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


@dataclass(frozen=True, kw_only=True)
class UnscentedTransform(SigmaPoints):
    """Unscented Transform"""
    model_fun: Callable
    fun_subtract_out: Callable = field(default=np.subtract, repr=False)
    fun_weighted_mean_out: Callable = field(
        default=partial(np.average, axis=0),
        repr=False
    )

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
        y_mvec = self.fun_weighted_mean_out(y_spts, weights=self.wm)
        # print(np.degrees(y_mvec))
        x_res = [self.fun_subtract_in(ix, m) for ix in x_spts]
        y_res = [self.fun_subtract_out(iy, y_mvec) for iy in y_spts]
        Pyy = sum([iw * np.outer(iy, iy) for iy, iw in zip(y_res, self.wc)])
        Pxy = sum([iw * np.outer(ix, iy)
                   for ix, iy, iw in zip(x_res, y_res, self.wc)])
        return y_mvec, Pyy, Pxy
