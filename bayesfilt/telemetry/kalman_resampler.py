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
from bayesfilt.filters import KalmanFilter
from bayesfilt.models import ConstantAcceleration1D, ConstantVelocity1D
from bayesfilt.models import LinearObservationModel
from .utils import get_bin_edges
from ._base_data import BaseClass


@dataclass
class ResamplerBase(BaseClass):
    """Base class for defining a resampler"""
    mm = None
    om = None
    kf = None

    def __repr__(self):
        out_str = self.mm.__str__() + '\n'
        out_str += self.om.__str__() + '\n'
        out_str += self.kf.__str__() + '\n'
        return out_str


class ConstantVelocityResampler(ResamplerBase):
    """Class for Constant Velocity based resampler"""

    def __init__(
        self,
        dt: float,
        error_strength: float,
        dim_name: str = ''
    ):
        # motion model
        super().__init__()
        self.mm = ConstantVelocity1D()
        self.mm.state_names = ['Position', 'Velocity']
        self.mm.state_names = [f'{ix}{dim_name}' for ix in self.mm.state_names]
        self.mm.dt = dt
        self.mm.phi['sigma'] = error_strength
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
        positions: ndarray,
        error_std: ndarray,
        start_state_std: list[float],
        object_id: int = 0,
        smoother: bool = True
    ):
        """Resample the track"""
        self.printit(f'Resampling track-{object_id}..')
        self.kf.objectid = object_id
        positions = np.atleast_1d(positions)
        times = np.atleast_1d(times)
        error_std = np.atleast_1d(error_std)
        start_vel = (positions[1] - positions[0]) / times[0]
        self.kf.m = [positions[0], start_vel]
        self.kf.P = np.diag(start_state_std)
        self.kf.initiate(t0=times[0])
        list_of_rmats = [np.diag([ix**2]) for ix in error_std]
        self.kf.filter(times[1:], positions[1:], list_of_rmats[1:])
        if smoother:
            self.kf.smoother()


class ConstantAccelerationResampler(ResamplerBase):
    """Class for Constant Acceleration based resampler"""

    def __init__(
        self,
        dt: float,
        error_strength: float,
        dim_name: str = ''
    ):
        # motion model
        super().__init__()
        self.mm = ConstantAcceleration1D()
        self.mm.state_names = ['Position', 'Velocity', 'Acceleration']
        self.mm.state_names = [f'{ix}{dim_name}' for ix in self.mm.state_names]
        self.mm.dt = dt
        self.mm.phi['sigma'] = error_strength
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
        positions: ndarray,
        error_std: ndarray,
        start_state_std: list[float],
        object_id: int = 0,
        smoother: bool = True
    ):
        """Resample the track"""
        self.printit(f'Resampling track-{object_id}..')
        self.kf.objectid = object_id
        positions = np.atleast_1d(positions)
        times = np.atleast_1d(times)
        error_std = np.atleast_1d(error_std)
        start_vel = (positions[1] - positions[0]) / times[0]
        self.kf.m = [positions[0], start_vel, 0.]
        self.kf.P = np.diag(start_state_std)
        self.kf.initiate(t0=times[0])
        list_of_rmats = [np.diag([ix**2]) for ix in error_std]
        self.kf.filter(times[1:], positions[1:], list_of_rmats[1:])
        if smoother:
            self.kf.smoother()


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
#         positions: ndarray,
#         error_std: ndarray,
#         start_state_std: list[float],
#         object_id: int = 0,
#         smoother: bool = True
#     ):
#         """Resample the track"""
#         self.printit(f'Resampling track-{object_id}..')
#         self.kf.objectid = object_id
#         positions = np.atleast_1d(positions)
#         times = np.atleast_1d(times)
#         error_std = np.atleast_1d(error_std)
#         start_vel = (positions[1] - positions[0]) / times[0]
#         if self.mm.nx == 3:
#             self.kf.m = [positions[0], start_vel, 0.]
#             self.kf.P = np.diag(start_state_std)
#         else:
#             self.kf.m = [positions[0], start_vel]
#             self.kf.P = np.diag(start_state_std)
#         self.kf.initiate(t0=times[0])
#         list_of_rmats = [np.diag([ix**2]) for ix in error_std]
#         self.kf.filter(times[1:], positions[1:], list_of_rmats[1:])
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
