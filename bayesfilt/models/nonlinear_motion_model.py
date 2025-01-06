"""Classes defining nonlinear motion models"""
# pylint: disable=invalid-name
from abc import abstractmethod
from dataclasses import dataclass
from numpy import ndarray
from ._base_model import StateSpaceModel


@dataclass(frozen=True, kw_only=True)
class NonlinearMotionModel(StateSpaceModel):
    """Base class for nonlinear motion model"""

    @abstractmethod
    def func_f(
        self,
        x: ndarray,
        dt: float,
    ) -> ndarray:
        """Model dynamics equation"""

    @abstractmethod
    def func_Q(
        self,
        x: ndarray,
        dt: float,
    ) -> ndarray:
        """Error covariance matrix"""
