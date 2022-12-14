# """Classes defining linear motion models"""
# from abc import abstractmethod
# from typing import Dict
# from numpy import ndarray
# from .motion_model import MotionModel


# class LinearMotionModel(MotionModel):
#     # pylint: disable=invalid-name
#     """Base class for linear motion model with additive Gaussain errors"""

#     def __init__(
#         self,
#         nx: int,
#         dt: float,
#         name: str = 'NonlinearMotionModel'
#     ) -> None:
#         super().__init__(nx=nx, dt=dt, nq=nx, name=name)

#     @abstractmethod
#     def update_matrices(self) -> None:
#         """Update system matrices"""

#     @abstractmethod
#     def f(
#         self,
#         x: ndarray,
#         q: ndarray | None,
#         u: ndarray | None
#     ) -> ndarray:
#         """Model dynamics equation"""

#     @abstractmethod
#     def compute_F(
#         self,
#         x: ndarray | None,
#         q: ndarray | None
#     ) -> ndarray:
#         """Get F matrix"""

#     @abstractmethod
#     def compute_G(
#         self,
#         x: ndarray | None,
#         q: ndarray | None
#     ) -> ndarray:
#         """Get G matrix"""

#     @abstractmethod
#     def compute_Q(
#         self,
#         x: ndarray | None,
#         q: ndarray | None
#     ) -> ndarray:
#         """Get Q matrix"""
