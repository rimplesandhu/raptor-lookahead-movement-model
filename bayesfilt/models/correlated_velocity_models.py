"""Classes defining various versions of Correlated velocity models"""
# pylint: disable=invalid-name
from numpy import ndarray
from copy import deepcopy
import numpy as np
import scipy.linalg as sl
from .linear_motion_model import LinearMotionModel
from .nonlinear_motion_model import NonlinearMotionModel


class CVM2D(LinearMotionModel):
    # pylint: disable=invalid-name
    """Class for Correlated velocity model in 1D"""

    def __init__(self):
        super().__init__(nx=4, name='CorrelatedVelocityModel2D')
        self._state_names = [
            'PositionX', 'PositionY', 'VelocityX', 'VelocityY'
        ]

    @property
    def phi_names(self):
        """Parameter names"""
        return ['eta', 'tau', 'omega', 'mu_x', 'mu_y']

    def update_matrices(self):
        """Update system matrices"""
        self.check_ready_to_deploy()
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
        vec_mu = np.array([self.phi['mu_x'], self.phi['mu_y']])
        temp_a = 2. * self.phi['eta']**2 / self.phi['tau']

        self.qbar = np.block([
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


class CVM3D(LinearMotionModel):
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

    @property
    def phi_names(self):
        """Parameter names"""
        return ['eta_hor', 'tau_hor', 'omega', 'sigma_mu_hor',
                'eta_ver', 'tau_ver', 'sigma_mu_ver']

    def update_matrices(self):
        """Update system matrices"""
        self.check_ready_to_deploy()

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
        ver_Ainv = self.phi['tau_ver']
        ver_R1 = ver_Ainv * (1. - np.exp(-ver_A * self.dt))
        ver_R2 = (self.phi['eta_ver']**2) * \
            (1. - np.exp(-2. * self.dt * ver_A))
        ver_c = 2. * self.phi['eta_ver']**2 / self.phi['tau_ver']
        ver_F = np.array([
            [1., ver_R1, self.dt - ver_R1],
            [0., 1 - ver_A * ver_R1, ver_A * ver_R1],
            [0., 0., 1.]
        ])
        ver_Q1 = ver_Ainv**2 * (ver_c * (self.dt - 2. * ver_R1) + ver_R2)
        ver_Q12 = ver_Ainv * (ver_c * ver_R1 - ver_R2)
        ver_Q = np.array([
            [ver_Q1 * 1., ver_Q12 * 1., 0.],
            [ver_Q12 * 1., ver_R2 * 1., 0.],
            [0., 0., self.phi['sigma_mu_ver']**2]
        ])

        # combined
        self._F = np.zeros((self.nx, self.nx))
        self._F[0:6, 0:6] = hor_F
        self._F[6:9, 6:9] = ver_F

        self._Q = np.zeros((self.nx, self.nx))
        self._Q[0:6, 0:6] = hor_Q
        self._Q[6:9, 6:9] = ver_Q


class CVM3D_NL(NonlinearMotionModel):
    """Class for CVM3D"""

    def __init__(self):
        super().__init__(nx=10, name='CVM3D_NL_V1')
        self._cvm3d = CVM3D()
        self._state_names = self._cvm3d.state_names + ['Omega']

    @ property
    def phi_names(self):
        """Parameter names"""
        return ['eta_hor', 'tau_hor', 'sigma_mu_hor', 'sigma_omega',
                'eta_ver', 'tau_ver', 'sigma_mu_ver']

    def _update_base_model(self, x):
        self.check_ready_to_deploy()
        self._cvm3d.dt = self.dt
        self._cvm3d.phi = {k: v for k,
                           v in self.phi.items() if k in self._cvm3d.phi_names}
        self._cvm3d.phi['Omega'] = deepcopy(x[9])
        self._cvm3d.update_matrices()

    def func_f(self, x: ndarray, u=None):
        """Model dynamics equation"""
        self._update_base_model(x)
        aug_F = np.eye(1)
        self._F = np.zeros((self.nx, self.nx))
        self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
        self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
        return self.F @ x

    def func_Q(self, x: ndarray):
        """Error covariance equation"""
        self._update_base_model(x)
        aug_Q = np.diag([
            self.phi['sigma_omega']**2,
        ])
        self._Q = np.zeros((self.nx, self.nx))
        self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
        self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
        return self.Q


class CVM3D_NL_2(NonlinearMotionModel):
    """Class for CVM3D"""

    def __init__(self):
        super().__init__(nx=11, name='CVM3D_NL_2')
        self._cvm3d = CVM3D()
        self._state_names = self._cvm3d.state_names + ['Omega', 'TauVer']

    @ property
    def phi_names(self):
        """Parameter names"""
        return ['eta_hor', 'tau_hor', 'sigma_mu_hor', 'sigma_omega',
                'eta_ver', 'sigma_tau_ver', 'sigma_mu_ver']

    def _update_base_model(self, x):
        self.check_ready_to_deploy()
        self._cvm3d.dt = self.dt
        self._cvm3d.phi = {k: v for k,
                           v in self.phi.items() if k in self._cvm3d.phi_names}
        self._cvm3d.phi['omega'] = deepcopy(x[9])
        self._cvm3d.phi['tau_ver'] = deepcopy(x[10])
        self._cvm3d.update_matrices()

    def func_f(self, x: ndarray, u=None):
        """Model dynamics equation"""
        self._update_base_model(x)
        aug_F = np.eye(2)
        self._F = np.zeros((self.nx, self.nx))
        self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
        self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
        return self.F @ x

    def func_Q(self, x: ndarray):
        """Error covariance equation"""
        self._update_base_model(x)
        aug_Q = np.diag([
            self.phi['sigma_omega']**2,
            1 * self.phi['sigma_tau_ver']**2
        ])
        self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
        self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
        return self.Q


class CVM3D_NL_3(NonlinearMotionModel):
    """Class for CVM3D"""

    def __init__(self):
        super().__init__(nx=11, name='CVM3D_NL_3')
        self._cvm3d = CVM3D()
        self._state_names = self._cvm3d.state_names + ['Omega', 'LogTauVer']

    @ property
    def phi_names(self):
        """Parameter names"""
        return ['eta_hor', 'tau_hor', 'sigma_mu_hor', 'sigma_omega',
                'eta_ver', 'sigma_log_tau_ver', 'sigma_mu_ver']

    def _update_base_model(self, x):
        self.check_ready_to_deploy()
        self._cvm3d.dt = self.dt
        self._cvm3d.phi = {k: v for k,
                           v in self.phi.items() if k in self._cvm3d.phi_names}
        self._cvm3d.phi['omega'] = deepcopy(x[9])
        self._cvm3d.phi['tau_ver'] = np.exp(deepcopy(x[10]))
        self._cvm3d.update_matrices()

    def func_f(self, x: ndarray, u=None):
        """Model dynamics equation"""
        self._update_base_model(x)
        aug_F = np.eye(2)
        self._F = np.zeros((self.nx, self.nx))
        self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
        self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
        return self.F @ x

    def func_Q(self, x: ndarray):
        """Error covariance equation"""
        self._update_base_model(x)
        aug_Q = np.diag([
            self.phi['sigma_omega']**2,
            1 * self.phi['sigma_log_tau_ver']**2
        ])
        self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
        self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
        return self.Q


def correct_omega(anglerate):
    if anglerate > np.pi:
        anglerate -= 2 * np.pi
    elif anglerate < -np.pi:
        anglerate += 2. * np.pi
    return anglerate


class CVM3D_NL_4(NonlinearMotionModel):
    """Class for CVM3D"""

    def __init__(self):
        super().__init__(nx=12, name='CVM3D_NL_4')
        self._cvm3d = CVM3D()
        self._state_names = self._cvm3d.state_names + \
            ['Omega', 'LogTauVer', 'LogTauHor']

    @ property
    def phi_names(self):
        """Parameter names"""
        return ['eta_hor', 'sigma_log_tau_hor', 'sigma_mu_hor', 'sigma_omega',
                'eta_ver', 'sigma_log_tau_ver', 'sigma_mu_ver']

    def _update_base_model(self, x):
        self.check_ready_to_deploy()
        self._cvm3d.dt = self.dt
        self._cvm3d.phi = {k: v for k,
                           v in self.phi.items() if k in self._cvm3d.phi_names}
        self._cvm3d.phi['omega'] = deepcopy(x[9])
        self._cvm3d.phi['tau_ver'] = np.exp(deepcopy(x[10]))
        self._cvm3d.phi['tau_hor'] = np.exp(deepcopy(x[11]))
        self._cvm3d.update_matrices()

    def func_f(self, x: ndarray, u=None):
        """Model dynamics equation"""
        self._update_base_model(x)
        aug_F = np.eye(3)
        self._F = np.zeros((self.nx, self.nx))
        self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
        self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
        return self.F @ x

    def func_Q(self, x: ndarray):
        """Error covariance equation"""
        self._update_base_model(x)
        aug_Q = np.diag([
            self.phi['sigma_omega']**2,
            self.phi['sigma_log_tau_ver']**2,
            self.phi['sigma_log_tau_hor']**2
        ])
        self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
        self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
        return self.Q

# class CVM3D_EXT1(LinearMotionModel):
#     # pylint: disable=invalid-name
#     """Class for Correlated velocity model in 1D"""

#     def __init__(self):
#         super().__init__(nx=12, name='CVM3D_Augmented')
#         self._cvm3d = CVM3D()
#         self._state_names = self._cvm3d.state_names + \
#             ['AglTransformed', 'OroTransformed', 'ThrTransformed']

#     @property
#     def phi_names(self):
#         """Parameter names"""
#         phi_base = ['eta_hor', 'tau_hor', 'omega', 'sigma_mu_hor',
#                     'eta_ver', 'tau_ver', 'sigma_mu_ver']
#         phi_aug = ['sigma_agl', 'sigma_oro', 'sigma_the']
#         return phi_base + phi_aug

#     def update_matrices(self):
#         """Update system matrices"""
#         self.check_ready_to_deploy()
#         self._cvm3d.dt = self.dt
#         self._cvm3d.phi = {k: v for k,
#                            v in self.phi.items() if k in self._cvm3d.phi_names}
#         self._cvm3d.update_matrices()

#         # covariates
#         aug_F = np.eye(3)
#         aug_Q = np.diag([
#             self.phi['sigma_agl']**2,
#             self.phi['sigma_oro']**2,
#             self.phi['sigma_the']**2
#         ])

#         # combined
#         self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
#         self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F

#         self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
#         self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q


class CVM3D_NL_EXT2(NonlinearMotionModel):
    """Class for CVM3D"""

    def __init__(self):
        super().__init__(nx=14, name='CVM3D_NL_EXT2')
        self._cvm3d = CVM3D()
        self._state_names = self._cvm3d.state_names + \
            ['Omega', 'AglT', 'OroT', 'ThrT', 'WspeedT']

    @ property
    def phi_names(self):
        """Parameter names"""
        phi_base = ['eta_hor', 'eta_ver', 'tau_hor', 'tau_ver']
        phi_aug = ['sigma_agl', 'sigma_oro', 'sigma_the', 'sigma_wspeed',
                   'sigma_mu_hor', 'sigma_mu_ver', 'sigma_omega']
        return phi_base + phi_aug

    def _update_base_model(self, x):
        self.check_ready_to_deploy()
        self._cvm3d.dt = self.dt
        self._cvm3d.phi = {k: v for k,
                           v in self.phi.items() if k in self._cvm3d.phi_names}
        self._cvm3d.phi['omega'] = deepcopy(x[9])
        self._cvm3d.update_matrices()

    def func_f(self, x: ndarray, u=None):
        """Model dynamics equation"""
        self._update_base_model(x)
        aug_F = np.eye(5)
        self._F = np.zeros((self.nx, self.nx))
        self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
        self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
        return self.F @ x

    def func_Q(self, x: ndarray):
        """Error covariance equation"""
        self._update_base_model(x)
        aug_Q = np.diag([
            self.phi['sigma_omega']**2,
            self.phi['sigma_agl']**2,
            self.phi['sigma_oro']**2,
            self.phi['sigma_the']**2,
            self.phi['sigma_wspeed']**2
        ])
        self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
        self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
        return self.Q


# class CVM3D_EXT3(NonlinearMotionModel):
#     """Class for CVM3D"""

#     def __init__(self):
#         super().__init__(nx=15, name='CVM3D_NL')
#         self._cvm3d = CVM3D()
#         self._state_names = self._cvm3d.state_names + \
#             ['AglT', 'OroT', 'ThrT',
#              'Omega', 'TauHor', 'TauVer']

#     @ property
#     def phi_names(self):
#         """Parameter names"""
#         phi_base = ['eta_hor', 'eta_ver']
#         phi_aug = ['sigma_agl', 'sigma_oro', 'sigma_the',
#                    'sigma_mu_hor', 'sigma_mu_ver',
#                    'sigma_omega', 'sigma_tau_hor', 'sigma_tau_ver']
#         return phi_base + phi_aug

#     def _update_base_model(self, x):
#         self.check_ready_to_deploy()
#         self._cvm3d.dt = self.dt
#         self._cvm3d.phi = {k: v for k,
#                            v in self.phi.items() if k in self._cvm3d.phi_names}
#         self._cvm3d.phi['omega'] = deepcopy(x[12])
#         self._cvm3d.phi['tau_hor'] = deepcopy(x[13])
#         self._cvm3d.phi['tau_ver'] = deepcopy(x[14])
#         # print(*self._cvm3d.phi.items())
#         self._cvm3d.update_matrices()

#     def func_f(self, x: ndarray, u=None):
#         """Model dynamics equation"""
#         self._update_base_model(x)
#         aug_F = np.eye(6)
#         self._F = np.zeros((self.nx, self.nx))
#         self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
#         self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
#         return self.F @ x

#     def func_Q(self, x: ndarray):
#         """Error covariance equation"""
#         self._update_base_model(x)
#         aug_Q = np.diag([
#             self.phi['sigma_agl']**2,
#             self.phi['sigma_oro']**2,
#             self.phi['sigma_the']**2,
#             self.phi['sigma_omega']**2,
#             self.phi['sigma_tau_hor']**2,
#             self.phi['sigma_tau_ver']**2
#         ])
#         self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
#         self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
#         return self.Q


# class CVM3D_NL_EXT4(NonlinearMotionModel):
#     """Class for CVM3D"""

#     def __init__(self):
#         super().__init__(nx=15, name='CVM3D_NL')
#         self._cvm3d = CVM3D()
#         self._state_names = self._cvm3d.state_names + \
#             ['AglT', 'OroT', 'ThrT',
#              'Omega', 'TauHor', 'TauVer']

#     @ property
#     def phi_names(self):
#         """Parameter names"""
#         phi_base = ['eta_hor', 'eta_ver']
#         phi_aug = [
#             'sigma_agl', 'sigma_oro', 'sigma_the',
#             'sigma_mu_hor', 'sigma_mu_ver',
#             'sigma_omega', 'sigma_tau_hor', 'sigma_tau_ver',
#             'par_driftz_agl', 'par_driftz_oro', 'par_driftz_thr'
#         ]
#         return phi_base + phi_aug

#     def _update_base_model(self, x):
#         self.check_ready_to_deploy()
#         self._cvm3d.dt = self.dt
#         self._cvm3d.phi = {k: v for k,
#                            v in self.phi.items() if k in self._cvm3d.phi_names}
#         self._cvm3d.phi['omega'] = deepcopy(x[12])
#         self._cvm3d.phi['tau_hor'] = deepcopy(x[13])
#         self._cvm3d.phi['tau_ver'] = deepcopy(x[14])
#         # print(*self._cvm3d.phi.items())
#         self._cvm3d.update_matrices()

#     def func_f(self, x: ndarray, u=None):
#         """Model dynamics equation"""
#         self._update_base_model(x)
#         aug_F = np.eye(6)
#         self._F = np.zeros((self.nx, self.nx))
#         self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
#         self._F[8, 8] += self.phi['par_driftz_agl'] * x[9]
#         self._F[8, 8] += self.phi['par_driftz_oro'] * x[9] * x[10]
#         self._F[8, 8] += self.phi['par_driftz_thr'] * x[9] * x[11]
#         self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
#         return self.F @ x

#     def func_Q(self, x: ndarray):
#         """Error covariance equation"""
#         self._update_base_model(x)
#         aug_Q = np.diag([
#             self.phi['sigma_agl']**2,
#             self.phi['sigma_oro']**2,
#             self.phi['sigma_the']**2,
#             self.phi['sigma_omega']**2,
#             self.phi['sigma_tau_hor']**2,
#             self.phi['sigma_tau_ver']**2
#         ])
#         self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
#         self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
#         return self.Q


# class CVM3D_NL_ALL(NonlinearMotionModel):
#     """Class for CVM3D"""

#     def __init__(self):
#         super().__init__(nx=17, name='CVM3D_NL_ALL')
#         self._cvm3d = CVM3D()
#         self._state_names = self._cvm3d.state_names + \
#             ['AglT', 'OroT', 'ThrT',
#              'Omega', 'TauHor', 'TauVer', 'EtaHor', 'EtaVer']

#     @ property
#     def phi_names(self):
#         """Parameter names"""
#         # phi_base = ['eta_hor', 'eta_ver']
#         phi_aug = ['sigma_agl', 'sigma_oro', 'sigma_the',
#                    'sigma_mu_hor', 'sigma_mu_ver',
#                    'sigma_omega', 'sigma_tau_hor', 'sigma_tau_ver',
#                    'sigma_eta_hor', 'sigma_eta_ver']
#         return phi_aug

#     def _update_base_model(self, x):
#         self.check_ready_to_deploy()
#         self._cvm3d.dt = self.dt
#         self._cvm3d.phi = {k: v for k,
#                            v in self.phi.items() if k in self._cvm3d.phi_names}
#         self._cvm3d.phi['omega'] = deepcopy(x[12])
#         self._cvm3d.phi['tau_hor'] = deepcopy(x[13])
#         self._cvm3d.phi['tau_ver'] = deepcopy(x[14])
#         self._cvm3d.phi['eta_hor'] = deepcopy(x[15])
#         self._cvm3d.phi['eta_ver'] = deepcopy(x[16])
#         # print(*self._cvm3d.phi.items())
#         self._cvm3d.update_matrices()

#     def func_f(self, x: ndarray, u=None):
#         """Model dynamics equation"""
#         self._update_base_model(x)
#         aug_F = np.eye(8)
#         self._F = np.zeros((self.nx, self.nx))
#         self._F[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.F
#         self._F[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_F
#         return self.F @ x

#     def func_Q(self, x: ndarray):
#         """Error covariance equation"""
#         self._update_base_model(x)
#         aug_Q = np.diag([
#             self.phi['sigma_agl']**2,
#             self.phi['sigma_oro']**2,
#             self.phi['sigma_the']**2,
#             self.phi['sigma_omega']**2,
#             self.phi['sigma_tau_hor']**2,
#             self.phi['sigma_tau_ver']**2,
#             self.phi['sigma_eta_hor']**2,
#             self.phi['sigma_eta_ver']**2
#         ])
#         self._Q = np.zeros((self.nx, self.nx))
#         self._Q[:self._cvm3d.nx, :self._cvm3d.nx] = self._cvm3d.Q
#         self._Q[self._cvm3d.nx:self.nx, self._cvm3d.nx:self.nx] = aug_Q
#         return self.Q
