""" Classes for defining linear motion models derived from basic models"""

import numpy as np
from numpy import ndarray
from .linear_motion_models import LinearMotionModel


class LinearModelA(LinearMotionModel):
    """Class for defining constant velocity model in 2d of a 2D object """

    def __init__(self):
        super().__init__(dim=6)
        self._cv2d = ConstantVelocityND(dof=2)
        self._rw3d = RandomWalkND(dof=2)

    def compute(
        self,
        delta_t: float,
        model_error_stds: ndarray,
        gamma_par: float = 1.
    ) -> None:
        """ Computes updated model matrices """
        model_error_stds = np.asarray(model_error_stds)
        assert model_error_stds.size == 4, 'Need 5 error stds for CV2DShape2D!'
        self._cv2d.compute(delta_t, model_error_stds[0:2], gamma_par)
        self._rw3d.compute(delta_t, model_error_stds[2:], gamma_par)
        upper_zeromat = np.zeros((self._cv2d.dim, self._rw3d.dim))
        self._F = np.block([[self._cv2d.F.copy(), upper_zeromat],
                            [upper_zeromat.T, self._rw3d.F.copy()]])
        self._Q = np.block([[self._cv2d.Q.copy(), upper_zeromat],
                            [upper_zeromat.T, self._rw3d.Q.copy()]])
