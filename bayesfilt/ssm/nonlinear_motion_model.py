"""Classes defining nonlinear motion models"""
from abc import abstractmethod
from numpy import ndarray
from .motion_model import MotionModel


class NonlinearMotionModel(MotionModel):
    # pylint: disable=invalid-name
    """Base class for nonlinear motion model"""

    def __init__(
        self,
        nx: int,
        name: str = 'NonlinearMotionModel'
    ) -> None:
        super().__init__(nx=nx, name=name)

    @abstractmethod
    def func_f(
        self,
        x: ndarray,
        u: ndarray | None
    ) -> ndarray:
        """Model dynamics equation"""

    @abstractmethod
    def func_Q(
        self,
        x: ndarray
    ) -> ndarray:
        """Error covariance matrix"""
