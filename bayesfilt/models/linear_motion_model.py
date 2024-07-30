"""Classes defining linear motion models"""
# pylint: disable=invalid-name
from abc import abstractmethod
import numpy as np
from .motion_model import MotionModel


class LinearMotionModel(MotionModel):
    """Base class for linear motion model with additive Gaussain errors"""

    def __init__(
        self,
        nx: int,
        name: str = 'LinearMotionModel'
    ) -> None:
        super().__init__(nx=nx, name=name)

    @abstractmethod
    def update_matrices(self, dt: float) -> None:
        """Update system matrices"""


class RandomWalk1D(LinearMotionModel):
    """Class for Random Walk 1D model """

    def __init__(self):
        super().__init__(nx=1, name='RandomWalk1D')

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigma']

    def update_matrices(self, dt: float) -> None:
        """Update system parameters"""
        self.check_ready_to_deploy()
        self._F[0, 0] = 1.
        self._Q[0, 0] = self.phi['sigma']**2 * dt**1 / 1


class RandomWalk(LinearMotionModel):
    """Class for Random Walk model in ND"""

    def __init__(self, nx: int):
        super().__init__(nx=nx, name=f'RandomWalk{nx}D')
        self._rw1d = RandomWalk1D()

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigmas']

    def update_matrices(self, dt: float) -> None:
        """Update system parameters"""
        self.check_ready_to_deploy()
        for i, v in enumerate(self.phi['sigmas']):
            self._rw1d.phi = {'sigma': v}
            self._rw1d.update_matrices(dt)
            self._F[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.F.copy()
            self._Q[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.Q.copy()


class ConstantVelocity1D(LinearMotionModel):
    """Class for constant velocity model in 1D"""

    def __init__(self):
        super().__init__(nx=2, name='ConstantVelocity1D')
        self.state_names = ['PositionX', 'VelocityX']

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigma']

    def update_matrices(self, dt: float) -> None:
        """Update system matrices"""
        self.check_ready_to_deploy()
        self._F[0, 0] = 1.
        self._F[0, 1] = dt
        self._F[1, 1] = 1.
        self._Q[0, 0] = 1. * dt**3 / 3
        self._Q[0, 1] = 1. * dt**2 / 2
        self._Q[1, 1] = dt
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
    def phi_names(self):
        """Parameter names"""
        return ['sigmas']

    def update_matrices(self, dt: float) -> None:
        """Update system parameters"""
        self.check_ready_to_deploy()
        for i, sigma in enumerate(self.phi['sigmas']):
            self._cv1d.phi = {'sigma': sigma}
            self._cv1d.update_matrices(dt=dt)
            self._F[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.F.copy()
            self._Q[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.Q.copy()


class ConstantAcceleration1D(LinearMotionModel):
    """Class for constant acceleration model in 1D"""

    def __init__(self):
        super().__init__(nx=3, name='ConstantAcceleration1D')
        self.state_names = ['PositionX', 'VelocityX', 'AccelerationX']

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigma']

    def update_matrices(self, dt: float) -> None:
        """Update system matrices"""
        self.check_ready_to_deploy()
        self._F = np.eye(self.nx)
        self._F[0, 1] = dt
        self._F[0, 2] = dt**2 / 2
        self._F[1, 2] = dt
        fac = 1.
        self._Q[0, 0] = fac * dt**5 / 20
        self._Q[0, 1] = fac * dt**4 / 8
        self._Q[0, 2] = fac * dt**3 / 6
        self._Q[1, 1] = fac * dt**3 / 3
        self._Q[1, 2] = fac * dt**2 / 2
        self._Q[2, 2] = dt
        self._Q = self.symmetrize(self.Q)
        self._Q *= self.phi['sigma']**2


class ConstantAcceleration(LinearMotionModel):
    """Class for constant acceleration model in N-dimension"""

    def __init__(self, dof: int):
        super().__init__(nx=int(3 * dof), name=f'ConstantAcceleration{dof}D')
        self._ca1d = ConstantAcceleration1D()
        self.state_names = [f'{ix}_{iy}' for iy in range(
            dof) for ix in self._ca1d.state_names]

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigmas']

    def update_matrices(self, dt: float) -> None:
        """Update system parameters"""
        self.check_ready_to_deploy()
        for i, sigma in enumerate(self.phi['sigmas']):
            self._ca1d.phi = {'sigma': sigma}
            self._ca1d.update_matrices(dt=dt)
            self._F[i * 3: (i + 1) * 3, i * 3: (i + 1)
                    * 3] = self._ca1d.F.copy()
            self._Q[i * 3: (i + 1) * 3, i * 3: (i + 1)
                    * 3] = self._ca1d.Q.copy()


class CV2DRW2D(LinearMotionModel):
    """Class for defining constant accn model in 2d of a 2D object """

    def __init__(self):
        super().__init__(nx=6, name='CV2D_Shape2D')
        self._motion = ConstantVelocity(dof=2)
        self._shape = RandomWalk(nx=2)
        self._state_names = [
            'PositionX', 'SpeedX',
            'PositionY', 'SpeedY',
            'Length', 'Width'
        ]

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigma_vx', 'sigma_vy', 'sigma_w', 'sigma_l']

    def update_matrices(self, dt: float) -> None:
        """ Computes updated model matrices """
        self.check_ready_to_deploy()
        # motion model
        self._motion.phi['sigmas'] = [
            self.phi['sigma_vx'], self.phi['sigma_vy']]
        self._motion.update_matrices(dt=dt)
        # shape model
        self._shape.phi['sigmas'] = [self.phi['sigma_w'], self.phi['sigma_l']]
        self._shape.update_matrices(dt=dt)

        upper_zeromat = np.zeros((self._motion.nx, self._shape.nx))
        self._F = np.block([[self._motion.F.copy(), upper_zeromat],
                            [upper_zeromat.T, self._shape.F.copy()]])
        self._Q = np.block([[self._motion.Q.copy(), upper_zeromat],
                            [upper_zeromat.T, self._shape.Q.copy()]])


class CA2DRW2D(LinearMotionModel):
    """Class for defining constant accn model in 2d of a 2D object """

    def __init__(self):
        super().__init__(nx=8, name='CA2D_Shape2D')
        self._motion = ConstantAcceleration(dof=2)
        self._shape = RandomWalk(nx=2)
        self._state_names = [
            'PositionX', 'SpeedX', 'AccnX',
            'PositionY', 'SpeedY', 'AccnY',
            'Length', 'Width'
        ]

    @property
    def phi_names(self):
        """Parameter names"""
        return ['sigma_ax', 'sigma_ay', 'sigma_w', 'sigma_l']

    def update_matrices(self, dt: float) -> None:
        """ Computes updated model matrices """
        self.check_ready_to_deploy()
        # motion model
        self._motion.phi['sigmas'] = [
            self.phi['sigma_ax'], self.phi['sigma_ay']]
        self._motion.update_matrices(dt=dt)
        # shape model
        self._shape.phi['sigmas'] = [self.phi['sigma_w'], self.phi['sigma_l']]
        self._shape.update_matrices(dt=dt)

        upper_zeromat = np.zeros((self._motion.nx, self._shape.nx))
        self._F = np.block([[self._motion.F.copy(), upper_zeromat],
                            [upper_zeromat.T, self._shape.F.copy()]])
        self._Q = np.block([[self._motion.Q.copy(), upper_zeromat],
                            [upper_zeromat.T, self._shape.Q.copy()]])
