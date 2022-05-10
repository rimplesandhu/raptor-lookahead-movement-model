"""Classes defining linear motion models"""
from numpy import ndarray
from .motion_model import MotionModel


class LinearMotionModel(MotionModel):
    # pylint: disable=invalid-name
    """Base class for linear motion model with additive Gaussain errors"""

    def __init__(self, nx: int, name: str) -> None:
        super().__init__(nx=nx, nq=nx, name=name)
        self._dof: int

    def f(
        self,
        x: ndarray,
        q: ndarray | None = None,
        u: ndarray | None = None
    ) -> ndarray:
        """Model dynamics equation"""
        x = self.vec_setter(x, self.nx)
        if self._F is None:
            self.raiseit('Need to initiate F matrix! Run update()')
        out_x = self._F @ x
        if q is not None:
            q = self.vec_setter(q, self.nx)
            out_x += q
        if u is not None:
            u = self.mat_setter(q, self.nx)
            out_x += u
        return out_x

    def compute_F(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> ndarray:
        """Get F matrix"""
        return self.F

    def compute_G(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> ndarray:
        """Get G matrix"""
        return self.G

    def compute_Q(
        self,
        x: ndarray | None = None,
        q: ndarray | None = None
    ) -> ndarray:
        """Get Q matrix"""
        return self.Q

    @property
    def dof(self) -> int:
        """Getter for degree of freedom"""
        return self._dof


class RandomWalk1D(LinearMotionModel):
    """Class for Random Walk 1D model """

    def __init__(self):
        super().__init__(nx=1, name='RW1D')
        self._dof = self._nx

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.float_setter(sigmas)
        self._initiate_matrices_to_identity()
        self._Q[0, 0] = self._sigmas**2 * self.dt**1 / 1


class RandomWalkND(LinearMotionModel):
    """Class for random walk model in ND"""

    def __init__(self, dof: int):
        self._dof = self.int_setter(dof)
        super().__init__(nx=self._dof, name=f'RW{self._dof}D')
        self._rw1d = RandomWalk1D()

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.vec_setter(sigmas, self._dof)
        self._initiate_matrices_to_identity()
        for i, sigma in enumerate(self.sigmas):
            self._rw1d.update(self.dt, sigma)
            self._F[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.F.copy()
            self._Q[i * 1:(i + 1) * 1, i * 1:(i + 1) * 1] = self._rw1d.Q.copy()


class ConstantVelocity1D(LinearMotionModel):
    """Class for constant velocity model in 1D"""

    def __init__(self):
        super().__init__(nx=2, name='CV1D')
        self._dof = 1
        self.labels = ['Position X', 'Speed X']

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.float_setter(sigmas)
        self._initiate_matrices_to_identity()
        self._F[0, 1] = self.dt
        self._Q[0, 0] = 1. * self.dt**3 / 3
        self._Q[0, 1] = 1. * self.dt**2 / 2
        self._Q[1, 1] = self.dt
        self._Q = self.symmetrize(self.Q)
        self._Q *= self.sigmas**2


class ConstantVelocityND(LinearMotionModel):
    """Class for constant velocity model in N-dimensions"""

    def __init__(self, dof: int):
        self._dof = self.int_setter(dof)
        super().__init__(nx=int(2 * self._dof), name=f'CV{self._dof}D')
        self._cv1d = ConstantVelocity1D()
        self._labels = []

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.vec_setter(sigmas, self._dof)
        self._initiate_matrices_to_identity()
        for i, sigma in enumerate(self.sigmas):
            self._cv1d.update(self.dt, sigma)
            self._F[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.F.copy()
            self._Q[i * 2:(i + 1) * 2, i * 2:(i + 1) * 2] = self._cv1d.Q.copy()
            self._labels += [istr + str(i + 1) for istr in self._cv1d.labels]


class ConstantAcceleration1D(LinearMotionModel):
    """Class for constant acceleration model in 1D"""

    def __init__(self):
        super().__init__(nx=3, name='CA1D')
        self._dof = 1
        self.labels = ['Position X', 'Speed X', 'Acceleration X']

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.float_setter(sigmas)
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
        self._Q *= self.sigmas**2


class ConstantAccelerationND(LinearMotionModel):
    """Class for constant acceleration model in N-dimension"""

    def __init__(self, dof: int):
        self._dof = self.int_setter(dof)
        super().__init__(nx=int(3 * self._dof), name=f'CA{self._dof}D')
        self._ca1d = ConstantAcceleration1D()
        self._labels = []

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: ndarray | None = None
    ) -> None:
        """Update system parameters and matrices"""
        self._dt = self.float_setter(dt)
        self._sigmas = self.vec_setter(sigmas, self._dof)
        self._initiate_matrices_to_identity()
        for i, sigma in enumerate(self.sigmas):
            self._ca1d.update(self.dt, sigma)
            self._F[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.F.copy()
            self._Q[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = self._ca1d.Q.copy()
            self._labels += [istr + str(i + 1) for istr in self._ca1d.labels]
