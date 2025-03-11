"""Base class for defining state space models"""
# pylint: disable=invalid-name
from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True)
class StateSpaceModel(ABC):
    """Base class for defining a state space model"""

    name: str
    nx: int
    xnames: list[str]

    def __post_init__(self):
        """post initiation function"""
        if self.xnames is not None:
            if len(self.xnames) != self.nx:
                self.raiseit(f'Len Mismatch: {len(self.xnames)}/nx{self.nx}')

    def printit(self, istr):
        """print it"""
        print(f'{self.__class__.__name__}: {istr}', flush=True)

    def raiseit(self, outstr: str = "", exception=ValueError) -> None:
        """Raise exception with the out string"""
        raise exception(f'{self.__class__.__name__}: {outstr}')

    def __repr__(self) -> str:
        """repr"""
        cls = self.__class__.__name__
        xnames = ','.join(self.xnames)
        return f'{cls}(name={self.name}, nx={self.nx}, xnames={xnames})'
    
    def check_state_names(self, xnames:list[str]):
        """Check if state names are valid"""
        for iname in xnames:
            if iname not in self.xnames:
                self.raiseit(f'Invalid state name: {iname}\nOptions:{self.xnames}')


@dataclass(frozen=True, kw_only=True)
class MotionModel(StateSpaceModel):
    """Base class for motion model"""


@dataclass(frozen=True, kw_only=True)
class ObservationModel(StateSpaceModel):
    """Base class for Observation model"""
