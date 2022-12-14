"""Classes defining linear motion models"""
from abc import abstractmethod
import numpy as np
from .motion_model import MotionModel
# pylint: disable=invalid-name


class LinearMotionModel(MotionModel):
    """Base class for linear motion model with additive Gaussain errors"""

    def __init__(
        self,
        nx: int,
        name: str = 'LinearMotionModel'
    ) -> None:
        super().__init__(nx=nx, nq=nx, name=name)

    @abstractmethod
    def update_matrices(self) -> None:
        """Update system matrices"""

    def __str__(self):
        out_str = super()._print_info()
        out_str += f'dt: {self.dt} second\n'
        out_str += f'qbar:\n {np.array_str(np.array(self.qbar), precision=3)}\n'
        out_str += f'F:\n {np.array_str(np.array(self.F), precision=3)}\n'
        out_str += f'G:\n {np.array_str(np.array(self.G), precision=3)}\n'
        out_str += f'Q:\n {np.array_str(np.array(self.Q), precision=3)}\n'
        return out_str


class RandomWalk1D(LinearMotionModel):
    """Class for Random Walk 1D model """

    def __init__(self):
        super().__init__(nx=1, name='RandomWalk1D')

    @property
    def phi_definition(self):
        """Parameter names"""
        return {'sigma': 1}

    def update_matrices(self) -> None:
        """Update system parameters"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        self._Q[0, 0] = self.phi['sigma']**2 * self.dt**1 / 1


class RandomWalk(LinearMotionModel):
    """Class for Random Walk model in ND"""

    def __init__(self, nx: int):
        super().__init__(nx=nx, name=f'RandomWalk{nx}D')
        self._rw1d = RandomWalk1D()

    @property
    def phi_definition(self):
        """Parameter names"""
        return {'sigmas': self.nx}

    def update_matrices(self) -> None:
        """Update system parameters"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        self._rw1d.dt = self.dt
        for i, sigma in enumerate(self.phi['sigmas']):
            self._rw1d.phi = {'sigma': sigma}
            self._rw1d.update_matrices()
            self._F[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.F.copy()
            self._Q[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.Q.copy()


class ConstantVelocity1D(LinearMotionModel):
    """Class for constant velocity model in 1D"""

    def __init__(self):
        super().__init__(nx=2, name='ConstantVelocity1D')
        self.state_names = ['PositionX', 'VelocityX']

    @property
    def phi_definition(self):
        """Parameter names"""
        return {'sigma': 1}

    def update_matrices(self) -> None:
        """Update system matrices"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        self._F[0, 1] = self.dt
        self._Q[0, 0] = 1. * self.dt**3 / 3
        self._Q[0, 1] = 1. * self.dt**2 / 2
        self._Q[1, 1] = self.dt
        self._Q = self.symmetrize(self.Q)
        self._Q *= self.phi['sigma']**2


class ConstantVelocity(LinearMotionModel):
    """Class for constant velocity model in N-dimensions"""

    def __init__(self, dof: int):
        super().__init__(nx=int(2 * dof), name=f'ConstantVelocity{dof}D')
        self._cv1d = ConstantVelocity1D()
        self.state_names = [f'{ix}_{iy}' for iy in range(
            dof) for ix in self._cv1d.state_names]

    @ property
    def phi_definition(self):
        """Parameter names"""
        return {'sigmas': int(self.nx // 2)}

    def update_matrices(self) -> None:
        """Update system parameters"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        self._cv1d.dt = self.dt
        for i, sigma in enumerate(self.phi['sigmas']):
            self._cv1d.phi = {'sigma': sigma}
            self._cv1d.update_matrices()
            self._F[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.F.copy()
            self._Q[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.Q.copy()


class ConstantAcceleration1D(LinearMotionModel):
    """Class for constant acceleration model in 1D"""

    def __init__(self):
        super().__init__(nx=3, name='ConstantAcceleration1D')
        self.state_names = ['PositionX', 'VelocityX', 'AccelerationX']

    @ property
    def phi_definition(self):
        """Parameter names"""
        return {'sigma': 1}

    def update_matrices(self) -> None:
        """Update system matrices"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        self._F[0, 1] = self.dt
        self._F[0, 2] = self.dt**2 / 2
        self._F[1, 2] = self.dt
        fac = 1.
        self._Q[0, 0] = fac * self.dt**5 / 20
        self._Q[0, 1] = fac * self.dt**4 / 8
        self._Q[0, 2] = fac * self.dt**3 / 6
        self._Q[1, 1] = fac * self.dt**3 / 3
        self._Q[1, 2] = fac * self.dt**2 / 2
        self._Q[2, 2] = self.dt
        self._Q = self.symmetrize(self.Q)
        self._Q *= self.phi['sigma']**2


class ConstantAcceleration(LinearMotionModel):
    """Class for constant acceleration model in N-dimension"""

    def __init__(self, dof: int):
        super().__init__(nx=int(3 * dof), name=f'ConstantAcceleration{dof}D')
        self._ca1d = ConstantAcceleration1D()
        self.state_names = [f'{ix}_{iy}' for iy in range(
            dof) for ix in self._ca1d.state_names]

    @ property
    def phi_definition(self):
        """Parameter names"""
        return {'sigmas': int(self.nx // 3)}

    def update_matrices(self) -> None:
        """Update system parameters"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        self._ca1d.dt = self.dt
        for i, sigma in enumerate(self.phi['sigmas']):
            self._ca1d.phi = {'sigma': sigma}
            self._ca1d.update_matrices()
            self._F[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.F.copy()
            self._Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.Q.copy()
