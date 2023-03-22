"""Classes for defining a nonlinear observation model"""
from abc import abstractmethod
from typing import List
import numpy as np
from numpy import ndarray
from .observation_model import ObservationModel


class NonlinearObservationModel(ObservationModel):
    # pylint: disable=invalid-name
    """Class for defining an observation model"""

    def __init__(
        self,
        nx: int,
        ny: int,
        ignore_inds_for_loglik: List[int] | None = None,
        name: str = 'NonlinearObservationModel',
        verbose: bool = False
    ) -> None:

        # model parameters
        super().__init__(
            nx=nx,
            ny=ny,
            ignore_inds_for_loglik=ignore_inds_for_loglik,
            name=name,
            verbose=verbose
        )

    @abstractmethod
    def h(
        self,
        x: ndarray | None,
        r: ndarray | None
    ) -> ndarray:
        """Measurement equation"""

    def __str__(self):
        out_str = super().__str__()
        if self.verbose:
            out_str += f'h: {self.h}\n'
            out_str += f'H:\n {np.array_str(np.array(self.H), precision=4)}\n'
            out_str += f'J:\n {np.array_str(np.array(self.J), precision=4)}\n'
            out_str += f'R:\n {np.array_str(np.array(self.R), precision=4)}\n'
        return out_str
