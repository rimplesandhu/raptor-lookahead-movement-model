# """Classes for defining an observation model"""
# from abc import abstractmethod
# from typing import List
# import numpy as np
# from numpy import ndarray
# from .state_space_model import StateSpaceModel


# class ObservationModel(StateSpaceModel):
#     # pylint: disable=invalid-name
#     """Class for defining an observation model"""

#     def __init__(
#         self,
#         nx: int,
#         ny: int,
#         name: str = 'ObsModel'
#     ) -> None:

#         # model parameters
#         super().__init__(nx=nx, name=name)
#         self._ny = self.int_setter(ny)  # dim of obs vector
#         self._obs_names = [f'y_{i}' for i in range(self.ny)]

#         # model matrices
#         self._H: ndarray | None = None  # Observation-State matrix
#         self._J: ndarray | None = None  # Error Jacobian matrix
#         self._R: ndarray | None = None  # Error covariance matrix

#     @abstractmethod
#     def h(
#         self,
#         x: ndarray | None,
#         r: ndarray | None
#     ) -> ndarray:
#         """Measurement equation"""

#     @abstractmethod
#     def compute_H(
#         self,
#         x: ndarray | None,
#         r: ndarray | None
#     ) -> ndarray:
#         """Get H matrix"""

#     @abstractmethod
#     def compute_J(
#         self,
#         x: ndarray | None,
#         r: ndarray | None
#     ) -> ndarray:
#         """Get J matrix"""

#     @property
#     def ny(self) -> int:
#         """Dimension of observation space """
#         return self._ny

#     @property
#     def H(self) -> ndarray:
#         """Measurement-State matrix"""
#         return self._H

#     @property
#     def J(self) -> ndarray:
#         """Measurement matrix"""
#         return self._J

#     @property
#     def R(self) -> ndarray:
#         """Measurement error covariance matrix"""
#         return self._R

#     @property
#     def obs_names(self) -> list[str]:
#         """Getter for labels"""
#         return self._obs_names

#     @obs_names.setter
#     def obs_names(self, in_list) -> None:
#         """Setter for labels"""
#         if len(in_list) != self.ny:
#             self.raiseit(f'Number of labels should be {self.ny}')
#         self._obs_names = in_list

#     def __str__(self):
#         out_str = super()._print_info()
#         out_str += f'Observation Dimension: {self._ny}\n'
#         out_str += f'H:\n {np.array_str(np.array(self._H), precision=4)}\n'
#         out_str += f'J:\n {np.array_str(np.array(self._J), precision=4)}\n'
#         out_str += f'R:\n {np.array_str(np.array(self._R), precision=4)}\n'
#         return out_str
