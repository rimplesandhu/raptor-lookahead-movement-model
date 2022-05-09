"""Classes for defining linear observation models """
import numpy as np
from numpy import ndarray
from .observation_model import ObservationModel


class LinearObservationModel(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining a linear observation model"""

    def __init__(
        self,
        nx: int,
        observed: dict,
        name: str = 'linear_obs'
    ) -> None:

        super().__init__(nx=nx, ny=len(observed), name=name,
                         observed=observed)
        self._H: ndarray = np.zeros((self._ny, self._nx))  # obs function
        for k, v in self.observed.items():
            self._H[int(k), int(v)] = 1.

    def update(
        self,
        sigmas: ndarray | None = None,
        R: ndarray | None = None,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        if R is not None:
            self._R = self.mat_setter(R, (self.ny, self.ny))
        else:
            self._sigmas = self.vec_setter(sigmas, self.ny)
            self._R = np.diag(self.sigmas**2)
        self._J = np.eye(self.ny)

    def h(
        self,
        x: ndarray | None,
        r: ndarray | None = None
    ) -> ndarray:
        """Measurement equation"""
        x = self.vec_setter(x, self.nx)
        out_y = self.H @ x
        if r is not None:
            r = self.vec_setter(r, self.ny)
            out_y += r
        return out_y

    def compute_H(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get H matrix"""
        return self.H

    def compute_J(
        self,
        x: ndarray | None = None,
        r: ndarray | None = None
    ) -> ndarray:
        """Get J matrix"""
        return self.J
