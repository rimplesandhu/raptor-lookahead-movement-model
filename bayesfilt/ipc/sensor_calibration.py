#!/usr/bin/env python
"""Traffic sensor class"""

# pylint: disable=invalid-name
# from dataclasses import dataclass, field, replace
import glob
import sys
import os
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from numpy import ndarray
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import cartopy.crs as ccrs
from tqdm import tqdm
from .traffic_sensor import TrafficSensor
from .utils import run_loop


class SensorCalibration:
    """Base class for sensor calibration"""

    def __init__(
        self,
        base_sensor: TrafficSensor,
        input_sensor: TrafficSensor,
    ):
        """Initialize"""
        self.base_sensor = deepcopy(base_sensor)
        self.input_sensor = deepcopy(input_sensor)

        self.bdf = self.base_sensor.df.copy()
        self.idf = self.input_sensor.df.copy()

        self.bdf_dur = self.base_sensor.df_duration.copy()
        self.idf_dur = self.input_sensor.df_duration.copy()

        self.pairs = []
        self.df = pd.DataFrame({})

    def __repr__(self) -> str:
        """Repr"""
        cls = self.__class__.__name__
        bname = self.base_sensor.name
        iname = self.input_sensor.name
        npairs = len(self.pairs)
        if self.df.empty:
            return f'{cls}(base={bname}, input={iname}, npairs={npairs})'
        else:
            tlap = np.round(self.df.time_overlap.sum(), 1)
            return f'{cls}(base={bname}, input={iname}, npairs={npairs}, overlap={tlap})'

    def gather_time_matched_pairs(
        self,
        min_time_overlap: float,
        min_base_lifespan: float | None = None,

    ):
        """Collect training data"""

        # check compatibiulity of parameters
        if min_base_lifespan is not None:
            if min_time_overlap >= min_base_lifespan:
                raise ValueError('min_time_overlap > min_base_lifespan')
        else:
            min_base_lifespan = min_time_overlap + 1.

        # get objects with lifespan greater than min_base_lifespan
        bbool = self.bdf_dur['Duration'] >= min_base_lifespan
        base_ids = self.bdf_dur.loc[bbool].index.values

        # clear previous history
        self.pairs.clear()
        self.df = pd.DataFrame({})

        # define progress bar
        tqdm_pbar = tqdm(
            iterable=base_ids,
            total=len(base_ids),
            desc='CalPaired',
            leave=True,
            ncols=80,
            file=sys.stdout,
            disable=False
        )

        # go over each object in base_sensor and findmatching pairs
        for idx in tqdm_pbar:
            # start, end times for the base object
            min_btime = self.bdf_dur.loc[idx, 'StartTime']
            max_btime = self.bdf_dur.loc[idx, 'EndTime']
            # print(idx, min_btime, max_btime)

            # get objects that overlap with base object
            ibool = self.idf['TimeElapsed'].between(min_btime, max_btime)
            groupby = self.idf.loc[ibool].groupby(['ObjectId'])
            odf = groupby['TimeElapsed'].max() - groupby['TimeElapsed'].min()
            # print(odf)

            ids = list(odf.loc[odf > min_time_overlap].index.values)
            durs = np.round(odf.loc[ids].values, 3)
            # print(durs)

            # assemble the identified pairs
            self.pairs += [(idx, ix, iy) for ix, iy in zip(ids, durs)]
            # print(ilist)

        # sort the list
        self.pairs = sorted(self.pairs, key=lambda ix: ix[2], reverse=True)
        # print(f'{len(self.pairs)} pairs for {len(base_ids)} objects')

    def get_rmse_speed(
        self,
        rmse_speed_threshold: float,
        max_time_duration: float = 60*60*6  # default to six hours
    ):
        """RMSE of speed between the pairs"""

        # define progress bar
        tqdm_pbar = tqdm(
            iterable=self.pairs,
            total=len(self.pairs),
            desc='CalMetric',
            leave=True,
            ncols=80,
            file=sys.stdout,
            disable=False
        )

        # iterate over each time matched pair
        _raw = []
        duration = 0.
        for ix, (bdx, idx, tlap) in enumerate(tqdm_pbar):
            min_btime = self.bdf_dur.loc[bdx, 'StartTime']
            max_btime = self.bdf_dur.loc[bdx, 'EndTime']
            bdfshort = self.bdf[self.bdf['ObjectId'] == bdx]

            idfshort = self.idf[self.idf['ObjectId'] == idx]

            min_itime = self.idf_dur.loc[idx, 'StartTime']
            max_itime = self.idf_dur.loc[idx, 'EndTime']

            start_t = max(min_itime, min_btime)
            end_t = min(max_itime, max_btime)

            ibool = idfshort['TimeElapsed'].between(start_t, end_t)
            bbool = bdfshort['TimeElapsed'].between(start_t, end_t)

            bspeed = bdfshort.loc[bbool, 'Speed'].values
            ispeed = idfshort.loc[ibool, 'Speed'].values
            rmse_speed = np.square(np.subtract(bspeed, ispeed))
            rmse_speed = np.sqrt(np.mean(rmse_speed))
            self.pairs[ix] = (bdx, idx, tlap, np.round(rmse_speed, 3))
            if rmse_speed < rmse_speed_threshold:
                _raw.append(dict(
                    base_obj=bdx,
                    input_obj=idx,
                    time_overlap=tlap,
                    rmse_speed=rmse_speed,
                    base_xpos=bdfshort.loc[bbool, 'PositionX'].values,
                    base_ypos=bdfshort.loc[bbool, 'PositionY'].values,
                    base_xspeed=bdfshort.loc[bbool, 'SpeedX'].values,
                    base_yspeed=bdfshort.loc[bbool, 'SpeedY'].values,
                    # base_xvar=bdfshort.loc[bbool, 'PositionX_Var'].values,
                    # base_yvar=bdfshort.loc[bbool, 'PositionY_Var'].values,
                    # base_dist=bdfshort.loc[bbool, 'Distance'].values,
                    input_xpos=idfshort.loc[ibool, 'PositionX'].values,
                    input_ypos=idfshort.loc[ibool, 'PositionY'].values,
                    # input_dist=idfshort.loc[ibool, 'Distance'].values,
                    input_xspeed=idfshort.loc[ibool, 'SpeedX'].values,
                    input_yspeed=idfshort.loc[ibool, 'SpeedY'].values,
                    # input_xvar=idfshort.loc[ibool, 'PositionX_Var'].values,
                    # input_yvar=idfshort.loc[ibool, 'PositionY_Var'].values,
                ))
                duration += tlap
            if duration > max_time_duration:
                break
        self.df = pd.DataFrame(_raw)
        istr = f'{self.base_sensor.name}-{self.input_sensor.name}'
        if not self.df.empty:
            tlap = np.round(self.df.time_overlap.sum(), 2)
        else:
            tlap = 0
        print(f'Found time/speed overlap of {tlap} sec b/w {istr}')

    def likelihood_function(
            self,
            pars: Tuple[float, float, float],
            speed_norm_factor: float = 1.
            # include_vars: bool = True,
            # scale_by_dist: bool = True
    ):
        """defines likelihood function for transformation"""
        if self.df.empty:
            raise ValueError('Need to assemble object pairs first!')

        theta_deg, tx, ty = pars
        tvec = np.array([tx, ty])
        theta = np.radians(theta_deg)
        Rmat = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])
        # Rmat = np.array([
        #     [np.cos(theta), np.sin(theta), 0.],
        #     [-np.sin(theta), np.cos(theta), 0.],
        #     [0., 0., 1.]
        # ])

        base_xy = np.vstack([
            np.concatenate(self.df['base_xpos'].ravel()),
            np.concatenate(self.df['base_ypos'].ravel())
        ])
        input_xy = np.vstack([
            np.concatenate(self.df['input_xpos'].ravel()),
            np.concatenate(self.df['input_ypos'].ravel())
        ])
        # base_xy_new = Rmat @ input_xy + tvec[:, np.newaxis]
        base_xy_new = Rmat @ (input_xy + tvec[:, np.newaxis])
        res_pos = np.subtract(base_xy, base_xy_new)
        res_norm_pos = np.array([np.dot(ires, ires) for ires in res_pos.T])

        base_xy = np.vstack([
            np.concatenate(self.df['base_xspeed'].ravel()),
            np.concatenate(self.df['base_yspeed'].ravel())
        ])
        input_xy = np.vstack([
            np.concatenate(self.df['input_xspeed'].ravel()),
            np.concatenate(self.df['input_yspeed'].ravel())
        ])
        # base_xy_new = Rmat @ input_xy + tvec[:, np.newaxis]
        base_xy_new = Rmat @ input_xy
        res_speed = np.subtract(base_xy, base_xy_new)
        res_norm_speed = np.array([np.dot(ires, ires) for ires in res_speed.T])

        # print(res_pos.shape, res_speed.shape, res.shape)

        # # if uncertianty in position needs to be included
        # if include_vars:
        #     base_vars = np.vstack([
        #         np.concatenate(self.df['base_xvar'].ravel()),
        #         np.concatenate(self.df['base_yvar'].ravel())
        #     ])
        #     Pinv = [np.diag(np.divide(1, ix)) for ix in base_vars.T]
        #     rnorm = [np.linalg.multi_dot([ires, iP, ires])
        #              for ires, iP in zip(res.T, Pinv)]
        # else:

        # # if distance from sensor needs to be includedssssss
        # if scale_by_dist:
        #     bdist = np.concatenate(self.df['base_dist'].ravel())
        #     idist = np.concatenate(self.df['input_dist'].ravel())
        #     rnorm = [ix*iy*iz/1e2 for ix, iy, iz in zip(rnorm, bdist, idist)]

        # return np.mean(res_norm_pos) + speed_norm_factor*np.mean(res_norm_speed)
        return np.mean(res_norm_pos + speed_norm_factor*res_norm_speed)


# def likelihood_function(pars):
#     theta_deg, tx, ty = pars
#     theta = np.radians(theta_deg)
#     norm = 0.
#     for _, idf, jdf in data_transform:
#         xd = idf[['PositionX', 'PositionY']].values.T
#         x = jdf[['PositionX', 'PositionY']].values.T
#         Pmat = idf[['PositionX_Var', 'PositionY_Var']].values
#         Pmat = [np.diag(np.divide(1, ix)) for ix in Pmat]
#         Rmat = np.array([[np.cos(theta), np.sin(theta)],
#                         [-np.sin(theta), np.cos(theta)]])
#         tvec = np.array([tx, ty])
#         xtr = Rmat @ x + tvec[:, np.newaxis]
#         res = list(np.subtract(xd, xtr).T)
#         # norm += np.sum([np.dot(ires,ires) for ires in res])
#         norm += np.sum([np.linalg.multi_dot([ires, iP, ires])
#                        for ires, iP in zip(res, Pmat)])
#     return norm/len(data_transform)
