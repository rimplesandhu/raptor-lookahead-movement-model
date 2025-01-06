""" Commonly used functions """
# pylint: disable=invalid-name
# from typing import Optional, Union, Dict, List
import numpy as np
from numpy import ndarray


def get_covariance_ellipse(icov: ndarray, fac: float):
    """Retruns width, height, and angle of the covariance ellipse"""
    vals, vecs = np.linalg.eigh(icov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2. * fac * np.sqrt(vals)
    return width, height, theta


# @dataclass(kw_only=True)
# class MotionModel(StateSpaceModel):
#     """Base class for defining a motion model"""

#     mat_F: ndarray

#     def __init__(
#         self,
#         nx: int,
#         name: str = 'MotionModel',
#         verbose: bool = False
#     ) -> None:

#         super().__init__(nx=nx, name=name, verbose=verbose)
#         self._F: ndarray = np.zeros((self.nx, self.nx))  # Transition matrix
#         self._Q: ndarray = np.zeros((self.nx, self.nx))  # Covariance matrix
#         self._G: ndarray = np.eye(self.nx)  # Error Jacobian matrix
#         self._qbar: ndarray = np.zeros((self.nx,))  # Error mean vector

#     def __repr__(self):
#         out_str = super().__repr__()
#         if self.verbose:
#             out_str += f'qbar: {np.array_str(np.array(self.qbar),
#                                              precision=3)}\n'
#             out_str += f'F:\n {np.array_str(np.array(self.F), precision=3)}\n'
#             out_str += f'G:\n {np.array_str(np.array(self.G), precision=3)}\n'
#             out_str += f'Q:\n {np.array_str(np.array(self.Q), precision=3)}\n'
#         return out_str

#     # def check_ready_to_deploy(self):
#     #     """Checks if all the model parameters are initiated"""
#     #     super().check_ready_to_deploy()
#     #     if self.dt is None:
#     #         self.raiseit('Need to assign dt')

#     @property
#     def qbar(self) -> float:
#         """Getter for time interval"""
#         return self._qbar

#     @qbar.setter
#     def qbar(self, in_list) -> float:
#         """Getter for time interval"""
#         self._qbar = self.valid_list(in_list, self.nx)

#     # @property
#     # def dt(self) -> float:
#     #     """Getter for time interval"""
#     #     return self._dt

#     # @dt.setter
#     # def dt(self, in_val: float) -> float:
#     #     """Getter for time interval"""
#     #     self._dt = self.scaler(in_val, dtype='float64')

#     @property
#     def F(self) -> ndarray:
#         """Getter for process matrix F"""
#         return self._F

#     @property
#     def G(self) -> ndarray:
#         """Getter for error jacobian matrix G"""
#         return self._G

#     @property
#     def Q(self) -> ndarray:
#         """Getter for error covariance matrix Q"""
#         return self._Q
