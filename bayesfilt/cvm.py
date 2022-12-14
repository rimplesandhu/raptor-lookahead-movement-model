"""Classes defining Correlated velocity models"""
import numpy as np
import scipy.linalg as sl
from .linear_motion_model import LinearMotionModel


class CorrelatedVelocityModel2D(LinearMotionModel):
    # pylint: disable=invalid-name
    """Class for Correlated velocity model in 1D"""

    def __init__(self):
        super().__init__(nx=4, name='CorrelatedVelocityModel2D')
        self._state_names = [
            'PositionX', 'PositionY', 'VelocityX', 'VelocityY'
        ]

    @ property
    def phi_definition(self):
        """Parameter names"""
        return {'eta': 1, 'tau': 1, 'omega': 1, 'mu': 2}

    def update_matrices(self):
        """Update system matrices"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()
        mat_A = np.array([
            [1 / self.phi['tau'], -self.phi['omega']],
            [self.phi['omega'], 1 / self.phi['tau']]
        ])
        det_A = 1 / (self.phi['tau']**2) + self.phi['omega']**2
        mat_Ainv = mat_A.T / det_A
        mat_T = np.eye(2) * self.dt

        mat_R1 = mat_Ainv @ (np.eye(2) - sl.expm(-mat_A * self.dt))
        mat_R2 = 1. - np.exp(-2. * self.dt / self.phi['tau'])
        mat_R2 = self.phi['eta']**2 * mat_R2 * np.eye(2)
        vec_mu = np.array(self.phi['mu'])
        temp_a = 2. * self.phi['eta']**2 / self.phi['tau']

        self.u = np.block([
            (mat_T - mat_R1) @ vec_mu,
            mat_A @ mat_R1 @ vec_mu
        ])
        self._F = np.block([
            [np.eye(2), mat_R1],
            [np.zeros((2, 2)), np.eye(2) - mat_A @ mat_R1]
        ])
        mat_Q1 = temp_a * mat_Ainv @ (mat_T - mat_R1 - mat_R1.T) @ mat_Ainv.T
        mat_Q1 += mat_Ainv @ mat_R2 @ mat_Ainv.T
        mat_Q12 = mat_Ainv @ (temp_a * mat_R1.T - mat_R2)
        mat_Q2 = mat_R2
        self._Q = np.block([[mat_Q1, mat_Q12], [mat_Q12.T, mat_Q2]])


class CorrelatedVelocityModel3D(LinearMotionModel):
    # pylint: disable=invalid-name
    """Class for Correlated velocity model in 1D"""

    def __init__(self):
        super().__init__(nx=9, name='CorrelatedVelocityModel2D')
        self._state_names = [
            'PositionX', 'PositionY',
            'VelocityX', 'VelocityY',
            'DriftX', 'DriftY',
            'PositionZ', 'VelocityZ', 'DriftZ'
        ]

    @ property
    def phi_definition(self):
        """Parameter names"""
        return {'eta_hor': 1, 'eta_ver': 1, 'tau_hor': 1, 'tau_ver': 1,
                'omega': 1, 'sigma_mu_hor': 1, 'sigma_mu_ver': 1}

    def update_matrices(self):
        """Update system matrices"""
        self._check_if_model_initiated_correctly()
        self._initiate_matrices_to_identity()

        # horizontal motion
        hor_A = np.array([
            [1 / self.phi['tau_hor'], -self.phi['omega']],
            [self.phi['omega'], 1 / self.phi['tau_hor']]
        ])
        hor_Adet = 1 / (self.phi['tau_hor']**2) + self.phi['omega']**2
        hor_Ainv = hor_A.T / hor_Adet
        hor_R1 = hor_Ainv @ (np.eye(2) - sl.expm(-hor_A * self.dt))
        hor_R2 = 1. - np.exp(-2. * self.dt / self.phi['tau_hor'])
        hor_R2 = self.phi['eta_hor']**2 * hor_R2 * np.eye(2)
        mat_T = np.eye(2) * self.dt
        hor_c = 2. * self.phi['eta_hor']**2 / self.phi['tau_hor']
        hor_F = np.block([
            [np.eye(2), hor_R1, mat_T - hor_R1],
            [np.zeros((2, 2)), np.eye(2) - hor_A @ hor_R1, hor_A @ hor_R1],
            [np.zeros((2, 2)), np.zeros((2, 2)), np.eye(2)]
        ])
        hor_Q1 = hor_c * hor_Ainv @ (mat_T - hor_R1 - hor_R1.T) @ hor_Ainv.T
        hor_Q1 += hor_Ainv @ hor_R2 @ hor_Ainv.T
        hor_Q12 = hor_Ainv @ (hor_c * hor_R1.T - hor_R2)
        hor_Q3 = self.phi['sigma_mu_hor']**2 * np.eye(2)
        hor_Q = np.block([
            [hor_Q1, hor_Q12, np.zeros((2, 2))],
            [hor_Q12.T, hor_R2, np.zeros((2, 2))],
            [np.zeros((2, 2)), np.zeros((2, 2)), hor_Q3]
        ])

        # vertical motion
        ver_A = 1 / self.phi['tau_ver']
        ver_Ainv = 1 / ver_A
        ver_R1 = ver_Ainv * (1. - np.exp(-ver_A * self.dt))
        ver_R2 = self.phi['eta_ver']**2 * (1. - np.exp(-2. * self.dt * ver_A))
        ver_c = 2. * self.phi['eta_ver']**2 / self.phi['tau_ver']
        ver_F = np.array([
            [1., ver_R1, self.dt - ver_R1],
            [0., 1 - ver_A * ver_R1, ver_A * ver_R1],
            [0., 0., 1.]
        ])
        ver_Q1 = ver_Ainv**2 * (ver_c * (self.dt - 2. * ver_R1) + ver_R2)
        ver_Q12 = ver_Ainv * (ver_c * ver_R1 - ver_R2)
        ver_Q = np.array([
            [ver_Q1, ver_Q12, 0.],
            [ver_Q12, ver_R2, 0.],
            [0., 0., self.phi['sigma_mu_ver']**2]
        ])

        # combined
        self._F = np.block([
            [hor_F, np.zeros((6, 3))],
            [np.zeros((3, 6)), ver_F]
        ])
        self._Q = np.block([
            [hor_Q, np.zeros((6, 3))],
            [np.zeros((3, 6)), ver_Q]
        ])
