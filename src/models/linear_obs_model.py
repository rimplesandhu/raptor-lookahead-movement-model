"""Classes for defining linear motion/observation models """
# pylint: disable=invalid-name
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from ._base_model import StateSpaceModel


class LinearObservationModel(StateSpaceModel):
    """Class for linear observation motion model with additive Gaussain errors"""

    def __init__(
            self,
            nx: int,
            ynames: list[int],
            xnames: list[str] = None,
            name:str='LinearObservationModel'
    ):
        """Initiatilization function"""
        super().__init__(
            nx=int(nx),
            name=name,
            xnames=xnames
        )

        self.check_state_names(xnames=ynames)
        self.xinds_obs = [i for i,e in enumerate(xnames) if e in ynames]
        self.ynames=ynames

        # H matrix
        self._Hmat: ndarray = np.zeros((self.ny, self.nx))  # obs function
        for k, v in enumerate(self.xinds_obs):
            self._Hmat[int(k), int(v)] = 1.

    def __repr__(self) -> str:
        """repr"""
        cls = self.__class__.__name__
        xnames = ','.join(self.xnames)
        istr = f'{cls}(name={self.name}, nx={self.nx}, ny={self.ny}, '
        ystr=','.join(self.ynames)
        istr += f'ynames=[{ystr}])'
        return istr

    @property
    def ny(self) -> int:
        """Dimension of observation space """
        return len(self.xinds_obs)

    @property
    def Hmat(self) -> ndarray:
        """Measurement-State matrix"""
        return self._Hmat
