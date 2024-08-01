""" Base class for defining metrics needed for Bayesian filtering """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name

from dataclasses import dataclass, fields
import numpy as np
from numpy import ndarray


@dataclass
class FilterMetrics:
    """Metrics"""
    XresNorm: float = None
    YresNorm: float = None
    NIS: float = None
    NEES: float = None
    LogLik: float = None

    def reset(self):
        """reset to none"""
        for field in fields(self):
            setattr(self, field.name, field.default)

    def compute(
        self,
        residual_x: ndarray | None = None,
        precision_x: ndarray | None = None,
        residual_y: ndarray | None = None,
        precision_y: ndarray | None = None
    ):
        """compute filter performance metrics"""
        # set to default values first
        self.reset()

        # is residual in data is provided
        if residual_y is not None:
            self.YresNorm = np.linalg.norm(residual_y)
            if precision_y is not None:
                self.NIS = np.linalg.multi_dot(
                    [residual_y.T, precision_y, residual_y])
                prec_det = np.linalg.det(precision_y)
                self.LogLik = len(residual_y) * np.log(2. * np.pi)
                self.LogLik += self.NIS - np.log(prec_det)
                self.LogLik *= -0.5

        # if residual in state is provided
        if residual_x is not None:
            self.XresNorm = np.linalg.norm(residual_x)
            if precision_x is not None:
                self.NEES = np.linalg.multi_dot(
                    [residual_x.T, precision_x, residual_x])

    def as_dict(self):
        """Provide current values as dict"""
        return {v.name: getattr(self, v.name) for v in fields(self)}
