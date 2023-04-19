""" Base class for Kalman Filtering based resampling of telemetry data """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
from typing import Callable
from functools import partial
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray
from bayesfilt.filters import KalmanFilter, KalmanFilterBase
from bayesfilt.filters import UnscentedKalmanFilter
from bayesfilt.models import MotionModel, ObservationModel
from bayesfilt.models import ConstantAcceleration1D, ConstantVelocity1D
from bayesfilt.models import LinearObservationModel, CVM3D_NL_4
from .utils import get_bin_edges
from ._base_data import BaseClass


@dataclass
class KalmanResampler(BaseClass):
    """Base class for defining a resampler"""
    dt: float
    flag: str = field(default='')
    smoother: bool = True
    mm: MotionModel = field(init=False)
    om: ObservationModel = field(init=False)
    kf: KalmanFilterBase = field(init=False)

    def __repr__(self):
        out_str = self.mm.__repr__() + '\n'
        out_str += self.om.__repr__() + '\n'
        out_str += self.kf.__repr__() + '\n'
        return out_str


class ConstantVelocityResampler(KalmanResampler):
    """Class for Constant Velocity based resampler"""

    def __init__(self, error_strength: float, **kwargs):
        KalmanResampler.__init__(self, **kwargs)

        # motion model
        self.mm = ConstantVelocity1D()
        self.mm.state_names = ['Position', 'Velocity']
        self.mm.state_names = [
            f'{ix}{self.flag}' for ix in self.mm.state_names]
        self.mm.dt = self.dt
        self.mm.phi = {'sigma': error_strength}
        self.mm.update_matrices()

        # observation model
        self.om = LinearObservationModel(
            nx=self.mm.nx,
            observed_state_inds=[0]
        )
        self.om.state_names = self.mm.state_names

        # kalman filter
        self.kf = KalmanFilter(
            nx=self.mm.nx,
            ny=self.om.ny,
            dt=self.mm.dt,
            mat_F=self.mm.F,
            mat_Q=self.mm.Q,
            mat_H=self.om.H,
            state_names=self.mm.state_names
        )

    def resample(
        self,
        times: ndarray,
        locs: ndarray,
        error_std: ndarray,
        start_state_std: list[float],
        object_id: int = 0,
    ):
        """Resample the track"""
        # self.printit(f'Resampling track-{object_id}..')
        self.kf.objectid = object_id
        locs = np.atleast_1d(locs)
        times = np.atleast_1d(times)
        error_std = np.atleast_1d(error_std)
        start_vel = (locs[1] - locs[0]) / times[0]
        self.kf.m = [locs[0], start_vel]
        self.kf.P = np.diag([ix**2 for ix in start_state_std])
        self.kf.initiate(t0=times[0])
        list_of_rmats = [np.diag([ix**2]) for ix in error_std]
        self.kf.filter(times[1:], locs[1:], list_of_rmats[1:])
        if self.smoother:
            self.kf.smoother()


class ConstantAccelerationResampler(KalmanResampler):
    """Class for Constant Acceleration based resampler"""

    def __init__(self, error_strength: float, **kwargs):
        KalmanResampler.__init__(self, **kwargs)

        # motion model
        self.mm = ConstantAcceleration1D()
        self.mm.state_names = ['Position', 'Velocity', 'Acceleration']
        self.mm.state_names = [
            f'{ix}{self.flag}' for ix in self.mm.state_names]
        self.mm.dt = self.dt
        self.mm.phi = {'sigma': error_strength}
        self.mm.update_matrices()

        # observation model
        self.om = LinearObservationModel(
            nx=self.mm.nx,
            observed_state_inds=[0]
        )
        self.om.state_names = self.mm.state_names

        # kalman filter
        self.kf = KalmanFilter(
            nx=self.mm.nx,
            ny=self.om.ny,
            dt=self.mm.dt,
            mat_F=self.mm.F,
            mat_Q=self.mm.Q,
            mat_H=self.om.H,
            state_names=self.mm.state_names
        )

    def resample(
        self,
        times: ndarray,
        locs: ndarray,
        error_std: ndarray,
        start_state_std: list[float],
        object_id: int = 0
    ):
        """Resample the track"""
        # self.printit(f'Resampling track-{object_id}..')
        self.kf.objectid = object_id
        locs = np.atleast_1d(locs)
        times = np.atleast_1d(times)
        error_std = np.atleast_1d(error_std)
        start_vel = (locs[1] - locs[0]) / times[0]
        self.kf.m = [locs[0], start_vel, 0.]
        self.kf.P = np.diag([ix**2 for ix in start_state_std])
        self.kf.initiate(t0=times[0])
        list_of_rmats = [np.diag([ix**2]) for ix in error_std]
        self.kf.filter(times[1:], locs[1:], list_of_rmats[1:])
        if self.smoother:
            self.kf.smoother()


class CorrelatedVelocityResampler(KalmanResampler):
    """Class for Constant Acceleration based resampler"""

    def __init__(self, phi: dict, **kwargs):
        KalmanResampler.__init__(self, **kwargs)

        # motion model
        self.mm = CVM3D_NL_4()
        self.mm.dt = self.dt
        self.mm.phi = phi
        # self.mm.update_matrices()

        # observation model
        self.om = LinearObservationModel(
            nx=self.mm.nx,
            observed_state_inds=[0, 1, 2, 3, 6, 7]
        )
        self.om.state_names = self.mm.state_names

        self.kf = UnscentedKalmanFilter(
            nx=self.mm.nx,
            ny=self.om.ny,
            dt=self.mm.dt,
            pars={'alpha': .0001, 'beta': 2.,
                  'kappa': -9, 'use_cholesky': True},
            fun_f=self.mm.func_f,
            fun_Q=self.mm.func_Q,
            mat_H=self.om.H,
            state_names=self.mm.state_names,
            verbose=False
        )

    def resample(
        self,
        dftrack,
    ):
        """Resample the track"""
        # self.printit(f'Resampling track-{object_id}..')
        self.kf.objectid = dftrack['TrackID'].iloc[0]
        start_state_dict = {
            'PositionX': (dftrack['PositionX'].iloc[0], 4.),
            'PositionY': (dftrack['PositionY'].iloc[0], 4.),
            'VelocityX': (dftrack['VelocityX'].iloc[0], 2.),
            'VelocityY': (dftrack['VelocityY'].iloc[0], 2.),
            'DriftX': (dftrack['VelocityX'].iloc[0] * 1, 2.),
            'DriftY': (dftrack['VelocityY'].iloc[0] * 1, 2.),
            'PositionZ': (dftrack['Altitude'].iloc[0], 5.),
            'VelocityZ': (dftrack['VelocityVer'].iloc[0], 2.),
            'DriftZ': (0., 1.),
            'Omega': (0., 0.05),  # 0.1 is 6 deg
            'LogTauVer': (-2, 0.5),
            'LogTauHor': (-2, 1.)
        }
        list_of_times = list(dftrack[['TrackTimeElapsed']].values[:, 0])
        self.kf.initiate_state(start_state_dict)
        self.kf.initiate(t0=list_of_times[0])
        observation_vars = {
            'PositionX': dftrack[['PositionX_var']].values.squeeze() * (2.5**2) * 1,
            'PositionY': dftrack[['PositionY_var']].values.squeeze() * (2.5**2) * 1,
            'VelocityX': dftrack[['VelocityX_var']].values.squeeze() * (2**2) * 1,
            'VelocityY': dftrack[['VelocityY_var']].values.squeeze() * (2**2) * 1,
            'Altitude': dftrack[['Altitude_var']].values.squeeze() * (2.5**2) * 1,
            'VelocityVer': dftrack[['VelocityVer_var']].values.squeeze() * (3**2) * 1
        }
        obs_vars = ['PositionX', 'PositionY', 'VelocityX',
                    'VelocityY', 'Altitude', 'VelocityVer']
        list_of_observation_covs = [
            np.diag(ix) for ix in np.array(list(observation_vars.values())).T]
        list_of_observations = list(dftrack[obs_vars].values)

        try:
            self.kf.filter(list_of_times[1:], list_of_observations[1:],
                           list_of_observation_covs[1:])
            if self.smoother:
                self.kf.smoother()
            # if np.amax(np.abs(kf.get_mean('Omega'))) > 2.:
            #     raise ValueError
            # print(f'{track_id}-good', flush=True)
        except Exception as e:
            # pass
            print(f'CVMresampler: {self.kf.objectid}-issue-{e}', flush=True)


# class KalmanTrackResampler:
#     """Class for applying linear Kalman Resapler"""

#     def __init__(
#         self,
#         dt: float,
#         model: str,
#         error_strength: float,
#         dim_name: str = ''
#     ):

#         # motion model
#         if model == 'CA':
#             self.mm = ConstantAcceleration1D()
#             self.mm.state_names = ['Position', 'Velocity', 'Acceleration']
#         elif model == 'CV':
#             self.mm = ConstantVelocity1D()
#             self.mm.state_names = ['Position', 'Velocity']
#         else:
#             print('Choose model as CA or CV!')
#         self.mm.dt = dt
#         self.mm.phi['sigma'] = error_strength
#         self.mm.update_matrices()
#         self.mm.state_names = [f'{ix}{dim_name}' for ix in self.mm.state_names]

#         # observation model
#         self.om = LinearObservationModel(
#             nx=self.mm.nx,
#             observed_state_inds=[0]
#         )
#         self.om.state_names = self.mm.state_names

#         # kalman filter
#         self.kf = KalmanFilter(
#             nx=self.mm.nx,
#             ny=self.om.ny,
#             dt=self.mm.dt,
#             mat_F=self.mm.F,
#             mat_Q=self.mm.Q,
#             mat_H=self.om.H,
#             state_names=self.mm.state_names
#         )

#     def __repr__(self):
#         out_str = self.mm.__str__() + '\n'
#         out_str += self.om.__str__() + '\n'
#         out_str += self.kf.__str__() + '\n'
#         return out_str

#     def resample(
#         self,
#         times: ndarray,
#         locs: ndarray,
#         error_std: ndarray,
#         start_state_std: list[float],
#         object_id: int = 0,
#         smoother: bool = True
#     ):
#         """Resample the track"""
#         self.printit(f'Resampling track-{object_id}..')
#         self.kf.objectid = object_id
#         locs = np.atleast_1d(locs)
#         times = np.atleast_1d(times)
#         error_std = np.atleast_1d(error_std)
#         start_vel = (locs[1] - locs[0]) / times[0]
#         if self.mm.nx == 3:
#             self.kf.m = [locs[0], start_vel, 0.]
#             self.kf.P = np.diag(start_state_std)
#         else:
#             self.kf.m = [locs[0], start_vel]
#             self.kf.P = np.diag(start_state_std)
#         self.kf.initiate(t0=times[0])
#         list_of_rmats = [np.diag([ix**2]) for ix in error_std]
#         self.kf.filter(times[1:], locs[1:], list_of_rmats[1:])
#         if smoother:
#             self.kf.smoother()

#     def printit(self, outstr: str = "", **kwargs) -> None:
#         """Print statement"""
#         print(f'{self.__class__.__name__}: {outstr}', flush=True, **kwargs)
#         # def kf_ca1d(
#         #     idftrack,
#         # ):
#         #     """Resample a track using Kalman Filter and Constant Acceleration model"""

#         #     track_id = idftrack['TrackID'].iloc[0]
#         #     list_of_times = list(idftrack[['TrackTimeElapsed']].values[:, 0])

#         #     mm.dt = 1.
#         #     om = LinearObservationModel(
#         #         nx=mm.nx,
#         #         observed_state_inds=[0]
#         #     )
#         #     kf_base = KalmanFilter(nx=mm.nx, ny=om.ny, dt=mm.dt)
#         #     kf_base.id = track_id
#         #     kf_base.H = om.H
#         #     list_of_dfs = []

#         #     # x movement
#         #     mm.phi['sigma'] = 0.8
#         #     mm.update_matrices()
#         #     ikf = deepcopy(kf_base)
#         #     ikf.F = mm.F
#         #     ikf.Q = mm.Q
#         #     ikf.state_names = ['PositionX', 'VelocityX', 'AccelerationX']
#         #     start_state_dict = {
#         #         'PositionX': (idftrack['PositionX'].iloc[0], 4.),
#         #         'VelocityX': (idftrack['VelocityX_TU'].iloc[0], 2.),
#         #         'AccelerationX': (0, 2.)
#         #     }
#         #     ikf.initiate_state_by_dict(list_of_times[0], start_state_dict)
#         #     list_of_obs = list(idftrack['PositionX'].values)
#         #     list_of_obs_std = list(idftrack['ErrorHDOP'].values * 2.5)
#         #     list_of_rmats = [np.diag([ix**2]) for ix in list_of_obs_std]
#         #     ikf.filter(list_of_times[1:], list_of_obs[1:], list_of_rmats[1:])
#         #     ikf.smoother()
#         #     list_of_dfs.append(ikf.df_smoother)

#         #     # y movement
#         #     mm.phi['sigma'] = 0.8
#         #     mm.update_matrices()
#         #     ikf = deepcopy(kf_base)
#         #     ikf.F = mm.F
#         #     ikf.Q = mm.Q
#         #     ikf.state_names = ['PositionY', 'VelocityY', 'AccelerationY']
#         #     start_state_dict = {
#         #         'PositionY': (idftrack['PositionY'].iloc[0], 4.),
#         #         'VelocityY': (idftrack['VelocityY_TU'].iloc[0], 2.),
#         #         'AccelerationY': (0, 2.)
#         #     }
#         #     ikf.initiate_state_by_dict(list_of_times[0], start_state_dict)
#         #     list_of_obs = list(idftrack['PositionY'].values)
#         #     list_of_obs_std = list(idftrack['ErrorHDOP'].values * 2.5)
#         #     list_of_rmats = [np.diag([ix**2]) for ix in list_of_obs_std]
#         #     ikf.filter(list_of_times[1:], list_of_obs[1:], list_of_rmats[1:])
#         #     ikf.smoother()
#         #     list_of_dfs.append(ikf.df_smoother)

#         #     # altitude
#         #     mm.phi['sigma'] = 0.2
#         #     mm.update_matrices()
#         #     ikf = deepcopy(kf_base)
#         #     ikf.F = mm.F
#         #     ikf.Q = mm.Q
#         #     ikf.state_names = ['Altitude', 'VelocityVer', 'AccelerationVer']
#         #     start_state_dict = {
#         #         'Altitude': (idftrack['Altitude'].iloc[0], 4.),
#         #         'VelocityVer': (idftrack['VelocityVer_FD'].iloc[0], 2.),
#         #         'AccelerationVer': (0, 2.)
#         #     }
#         #     ikf.initiate_state_by_dict(list_of_times[0], start_state_dict)
#         #     list_of_obs = list(idftrack['Altitude'].values)
#         #     list_of_obs_std = list(idftrack['ErrorVDOP'].values * 4.5)
#         #     list_of_rmats = [np.diag([ix**2]) for ix in list_of_obs_std]
#         #     ikf.filter(list_of_times[1:], list_of_obs[1:], list_of_rmats[1:])
#         #     ikf.smoother()
#         #     list_of_dfs.append(ikf.df_smoother)

#         #     # merge all
#         #     sdf = pd.concat(list_of_dfs, axis=1, join='outer')
#         #     # sdf = sdf.groupby(level=0, dropna=False).sum()
#         #     sdf = sdf.loc[:, ~sdf.columns.duplicated()]
#         #     # cols_to_drop = [ix for ix in sdf.columns if '_var' in ix]
#         #     cols_to_drop = [ix for ix in sdf.columns if 'Smoother' in ix]
#         #     sdf.drop(columns=cols_to_drop, inplace=True)
#         #     sdf['TimeUTC'] = pd.date_range(
#         #         start=idftrack['TimeUTC'].iloc[0],
#         #         periods=len(sdf),
#         #         freq=str(mm.dt) + "S"
#         #     )
#         #     sdf['TimeLocal'] = pd.date_range(
#         #         start=idftrack['TimeLocal'].iloc[0],
#         #         periods=len(sdf),
#         #         freq=str(mm.dt) + "S"
#         #     )
#         #     sdf['TrackPointCount'] = np.array([len(sdf)] * len(sdf)).astype('int32')
#         #     sdf['TrackFirstPoint'] = False
#         #     sdf.loc[0, 'TrackFirstPoint'] = True
#         #     sdf['TrackLastPoint'] = False
#         #     sdf['TrackDuration'] = np.array(
#         #         [mm.dt * len(sdf)] * len(sdf)).astype('float32')
#         #     sdf.loc[len(sdf) - 1, 'TrackLastPoint'] = True
#         #     sdf['AnimalID'] = [idftrack['AnimalID'].iloc[0]] * len(sdf)
#         #     sdf['Group'] = pd.Series(
#         #         [idftrack['Group'].iloc[0]] * len(sdf),
#         #         dtype="category"
#         #     )
#         #     sdf['Age'] = pd.Series(
#         #         [idftrack['Age'].iloc[0]] * len(sdf),
#         #         dtype="category"
#         #     )
#         #     sdf['Sex'] = pd.Series(
#         #         [idftrack['Sex'].iloc[0]] * len(sdf),
#         #         dtype="category"
#         #     )
#         #     return sdf
