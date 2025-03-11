"""Classes defining linear motion models"""
# pylint: disable=invalid-name
from abc import abstractmethod
import numpy as np
from scipy.linalg import block_diag
from numpy import ndarray
from dataclasses import dataclass
from ._base_model import StateSpaceModel


@dataclass(frozen=True, kw_only=True)
class LinearMotionModel(StateSpaceModel):
    """Base class for linear motion model with additive Gaussain errors"""

    @abstractmethod
    def func_Fmat(self, dt: float) -> ndarray:
        """Dynamics transition matrix"""

    @abstractmethod
    def func_Qmat(self, dt: float) -> ndarray:
        """Model error covariance matrix"""



class RandomWalk(LinearMotionModel):
    """Class for Random Walk model """

    def __init__(self, nx: int, sigma: ndarray, xnames: list[str] = None):
        """Initiatiolization for ranom walk model"""
        _names = [f'X{i}' for i in range(nx)]
        super().__init__(
            nx=int(nx),
            name=f'RandomWalk-{nx}D',
            xnames=_names if xnames is None else xnames
        )
        if len(sigma) != self.nx:
            self.raiseit(f'Mismatch:sigma{len(sigma)}/nx{self.nx}')
        self.sigma = sigma

    def func_Fmat(self, dt: float = None) -> ndarray:
        """Dynamics transition matrix"""
        return np.eye(self.nx, self.nx)

    def func_Qmat(self, dt: float) -> ndarray:
        """Dynamics transition matrix"""
        return np.diag(np.array(self.sigma)**2)*dt


class ConstantVelocity(LinearMotionModel):
    """Class for constant velocity model"""

    def __init__(self, dof: int, sigma_speed: ndarray, xnames: list[str] = None):
        """Initiatialization"""
        self.dof = int(dof)
        self.sigma_speed = np.atleast_1d(sigma_speed)
        if self.dof != len(self.sigma_speed):
            self.raiseit(
                f'len(sigma)={len(self.sigma_speed)},len(nx)={self.dof}')

        def snames(i): return [f'Position{i}', f'Speed{i}']
        _names = np.hstack([snames(i) for i in range(self.dof)])

        super().__init__(
            nx=int(2*dof),
            name=f'ConstantVelocity{dof}D',
            xnames=_names if xnames is None else xnames
        )

    def func_Fmat(self, dt: float = None) -> ndarray:
        """Dynamics transition matrix"""
        _mat = np.eye(2)
        _mat[0, 1] = dt
        return np.kron(np.eye(self.dof, dtype=int), _mat)

    def func_Qmat(self, dt: float = None) -> ndarray:
        """Model error covariance matrix"""
        _mat = np.zeros((2, 2))
        _mat[0, 1] = 1. * dt**2 / 2
        _mat[1, 0] = 1. * dt**2 / 2
        _mat[0, 0] = 1. * dt**3 / 3
        _mat[1, 1] = dt
        return np.kron(np.diag(self.sigma_speed**2), _mat)

    def __repr__(self) -> str:
        """repr"""
        return StateSpaceModel.__repr__(self)


class ConstantAcceleration(LinearMotionModel):
    """Class for constant acceleration model"""

    def __init__(self, dof: int, sigma_accn: ndarray, xnames: list[str] = None):
        """Initiatiolization function"""
        self.dof = int(dof)
        if self.dof != len(sigma_accn):
            self.raiseit(f'Mismatch:sigma{len(sigma_accn)}/nx{self.dof}')
        self.sigma_accn = np.array(sigma_accn)

        def snames(i): return [f'Position{i}', f'Speed{i}', f'Accn{i}']
        _names = np.hstack([snames(i) for i in range(self.dof)])
        super().__init__(
            nx=int(3*dof),
            name=f'ConstantAcceleration{dof}D',
            xnames=_names if xnames is None else xnames
        )

    def func_Fmat(self, dt: float = None) -> ndarray:
        """Dynamics transition matrix"""
        _mat = np.eye(3)
        _mat[0, 1] = dt
        _mat[0, 2] = dt**2/2
        _mat[1, 2] = dt
        return np.kron(np.eye(self.dof, dtype=int), _mat)

    def func_Qmat(self, dt: float = None) -> ndarray:
        """Model error covariance matrix"""
        _mat = np.zeros((3, 3))

        # off-diagonal
        _mat[0, 1] = dt**4 / 8
        _mat[0, 2] = dt**3 / 6
        _mat[1, 2] = dt**2 / 2
        _mat += _mat.T

        # diagonal
        _mat[0, 0] = dt**5 / 20
        _mat[1, 1] = dt**3 / 3
        _mat[2, 2] = dt

        return np.kron(np.diag(self.sigma_accn**2), _mat)

    def __repr__(self) -> str:
        """repr"""
        return StateSpaceModel.__repr__(self)


class CV_RW(LinearMotionModel):
    """Class for combining ConstantVelocityand RandomWalk models"""

    def __init__(
            self,
            dof_cv: int,
            sigma_speed: ndarray,
            dof_rw: int,
            sigma_rw: ndarray,
            xnames: list[str]
    ):
        """Initialization function"""
        self.cv_model = ConstantVelocity(dof=dof_cv, sigma_speed=sigma_speed)
        self.rw_model = RandomWalk(nx=dof_rw, sigma=sigma_rw)
        super().__init__(
            nx=self.cv_model.nx+self.rw_model.nx,
            name=f'CV{dof_cv}D+RW{dof_rw}D',
            xnames=list(xnames)
        )

    def func_Fmat(self, dt: float) -> ndarray:
        return block_diag(
            self.cv_model.func_Fmat(dt=dt),
            self.rw_model.func_Fmat())

    def func_Qmat(self, dt: float) -> ndarray:
        return block_diag(
            self.cv_model.func_Qmat(dt=dt),
            self.rw_model.func_Qmat(dt=dt)
        )


class CA_RW(LinearMotionModel):
    """Class for combining ConstantAcceleration and RandomWalk models"""

    def __init__(
            self,
            dof_ca: int,
            sigma_accn: ndarray,
            dof_rw: int,
            sigma_rw: ndarray,
            xnames: list[str]
    ):
        """Initialization function"""
        self.ca_model = ConstantAcceleration(dof=dof_ca, sigma_accn=sigma_accn)
        self.rw_model = RandomWalk(nx=dof_rw, sigma=sigma_rw)
        super().__init__(
            nx=self.ca_model.nx+self.rw_model.nx,
            name=f'CA{dof_ca}D+RW{dof_rw}D',
            xnames=list(xnames)
        )

    def func_Fmat(self, dt: float) -> ndarray:
        return block_diag(
            self.ca_model.func_Fmat(dt=dt),
            self.rw_model.func_Fmat()
        )

    def func_Qmat(self, dt: float) -> ndarray:
        return block_diag(
            self.ca_model.func_Qmat(dt=dt),
            self.rw_model.func_Qmat(dt=dt)
        )
