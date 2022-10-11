""" Classes for defining linear motion models derived from basic models"""

import numpy as np
from numpy import ndarray
from .linear_motion_models import *


class CA2D_Shape2D(LinearMotionModel):
    """Class for defining constant accn model in 2d of a 2D object """

    def __init__(self):
        super().__init__(nx=8, name='CA2D_Shape2D')
        self._motion = ConstantAccelerationND(dof=2)
        self._shape = RandomWalkND(dof=2)
        self._labels = ['X', 'SpeedX', 'AccnX', 'Y', 'SpeedY', 'AccnY',
                        'Length', 'Width']

    def update(
        self,
        dt: float,
        sigmas: ndarray,
        w: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        model_error_stds = np.asarray(sigmas)
        assert model_error_stds.size == 4, 'Need 4 error stds for CV2D_Shape2D!'
        self._motion.update(dt, model_error_stds[[0, 1]], sigmas)
        self._shape.update(dt, model_error_stds[[2, 3]], sigmas)
        upper_zeromat = np.zeros((self._motion.nx, self._shape.nx))
        self._F = np.block([[self._motion.F.copy(), upper_zeromat],
                            [upper_zeromat.T, self._shape.F.copy()]])
        self._Q = np.block([[self._motion.Q.copy(), upper_zeromat],
                            [upper_zeromat.T, self._shape.Q.copy()]])
