""" Base class for defining variable tracker needed for Bayesian filtering """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name

from dataclasses import dataclass, fields
from copy import deepcopy
from numpy import ndarray


@dataclass
class FilterVariables:
    """Variable tracker"""

    # vectors and matrices
    m: ndarray | None = None  # State mean vector
    P: ndarray | None = None  # State covariance matrix
    y: ndarray | None = None  # observation vector
    R: ndarray | None = None  # obs error cov matrix

    # residual vectors
    mres: ndarray | None = None  # state residual
    yres: ndarray | None = None  # data/obs residual

    # precision matrices
    Pinv: ndarray | None = None  # state precision matrix
    Sinv: ndarray | None = None  # obs precision matrix (not cond on state)

    # time related
    t: float | None = None  # time
    t_start: float | None = None
    t_last_update: float | None = None
    flag: str | None = None

    def is_initiated(self):
        """check if initiated"""
        if self.t_start is not None:
            if self.t_start == self.t:
                if self.m is not None:
                    if self.P is not None:
                        return True
        return False

    def reset(self):
        """reset to none"""
        for field in fields(self):
            setattr(self, field.name, field.default)

    @ property
    def lifespan_to_last_update(self) -> float:
        """Returns the time duration of existence till the last update"""
        return self.t_last_update - self.t_start

    @ property
    def lifespan_to_last_forecast(self) -> float:
        """Returns the time duration of existence till the last update"""
        return self.t - self.t_start

    # @ property
    # def y(self) -> ndarray:
    #     """Observation vector"""
    #     return self._y

    # @ property
    # def m(self) -> ndarray:
    #     """State mean vector"""
    #     return self._m

    # @ property
    # def P(self) -> ndarray:
    #     """State covariance matrix"""
    #     return self._P

    # @ property
    # def R(self) -> ndarray:
    #     """Observation error covariance matrix"""
    #     return self._R

    # @ property
    # def get_time_elapsed(self) -> ndarray:
    #     """Get time elapsed"""
    #     return self.df[self.time_colname].values

    # @ property
    # def start_time(self) -> ndarray:
    #     """Get time elapsed"""
    #     return self._start_time

    # @ m.setter
    # def m(self, in_vec: ndarray) -> None:
    #     """Setter for m vector"""
    #     self._m = validate_array(in_vec, self.nx, return_array=True)

    # @ P.setter
    # def P(self, in_mat: ndarray | None) -> None:
    #     """Setter for m vector"""
    #     self._P = validate_array(in_mat, (self.nx, self.nx), return_array=True)

    # @ R.setter
    # def R(self, in_mat: ndarray | None) -> None:
    #     """Setter for m vector"""
    #     if in_mat is not None:
    #         self._R = validate_array(in_mat, (self.ny, self.ny),
    #                                  return_array=True)
    #     else:
    #         self._R = None

    # @ y.setter
    # def y(self, in_vec: ndarray | None) -> None:
    #     """Setter for m vector"""
    #     if in_vec is not None:
    #         self._y = validate_array(in_vec, self.ny, return_array=True)
    #     else:
    #         self._y = None
