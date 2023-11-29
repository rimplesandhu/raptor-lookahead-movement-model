"""Classes defining nonlinear motion models"""
# pylint: disable=invalid-name
from abc import abstractmethod
from numpy import ndarray
from .motion_model import MotionModel


class NonlinearMotionModel(MotionModel):

    """Base class for nonlinear motion model"""

    def __init__(
        self,
        nx: int,
        name: str = 'NonlinearMotionModel',
        verbose: bool = False
    ) -> None:
        super().__init__(nx=nx, name=name, verbose=verbose)

    @abstractmethod
    def func_f(
        self,
        x: ndarray,
        u: ndarray | None,
        dt: ndarray | None
    ) -> ndarray:
        """Model dynamics equation"""

    @abstractmethod
    def func_Q(
        self,
        x: ndarray,
        u: ndarray | None,
        dt: ndarray | None
    ) -> ndarray:
        """Error covariance matrix"""
