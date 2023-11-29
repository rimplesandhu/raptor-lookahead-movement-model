#!/usr/bin/env python
"""Multisensor fusion engine class"""

# pylint: disable=invalid-name
import time
import datetime
from copy import deepcopy
from typing import List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from shapely.geometry import MultiPolygon
from bayesfilt.filters import KalmanFilterBase
from .traffic_sensor import TrafficSensor
from bayesfilt.models import ObservationModel


@dataclass
class ObjectLifespanManager:
    """Lifespan Management settings"""
    loglik_threshold: float
    pred_lifespan: float
    min_lifespan: float
    vehicle_lanes: MultiPolygon


class MultisensorFusionEngine:
    """Mult sensor fusion setup"""

    def __init__(
        self,
        kf_base: KalmanFilterBase,
        lifespan_manager: ObjectLifespanManager,
        name: str = 'NREL IPC Fusion Engine',
        verbose: bool = False
    ):
        self.name = name
        self.kf_base = kf_base
        self.olm = lifespan_manager
        self.verbose = verbose
        self._history_kf = {}
        self._past_objects = []
        self._cur_sensor = None
        self._cur_time = None
        self._cur_obs = None
        self._cur_om = None
        self._df = None
        self._logliks = {}
        self._quality_col = 'TrackQuality'

    def _initiate_new_object(self, at_id, at_mean):
        """Initiates new object id"""
        if self.verbose:
            print(f'Started {at_id} at {np.around(self._cur_time,3)}s')
        self._history_kf[at_id] = deepcopy(self.kf_base)
        self._history_kf[at_id].objectid = at_id
        self._history_kf[at_id].initiate(t0=self._cur_time, m0=at_mean)
        # if object_lengths is not None:
        #     len_idx = kf_base.state_names.index('Length')
        #     start_state_mean[len_idx] = object_lengths['Vehicle']

    def _kill_these_objects(self, these_ids: List[int]):
        """Kill/rmeove this object"""
        for this_id in list(these_ids):
            if self.verbose:
                print(
                    f'Killed {this_id} at {np.around(self._cur_time,3)}s')
            del self._history_kf[this_id]

    def _assemble_df(self):
        """Assemble the df with fused object list"""
        list_of_dfs = [self._history_kf[ix].df for ix in list(
            self._history_kf.keys())]
        if list_of_dfs:
            self._df = pd.concat(list_of_dfs)
            float64_cols = list(self._df.select_dtypes(include='float64'))
            self._df[float64_cols] = self._df[float64_cols].astype('float32')
        else:
            self._df = pd.DataFrame({})
        # self._df.drop(columns=['Observation', 'Training', 'FilterNEES',
        #                        'FilterNIS'], inplace=True, errors='ignore')
        # self._df['Time'] = [start_datetime + datetime.timedelta(
        #     milliseconds=1000 * ix) for ix in self._df['TimeElapsed']]
        # self._df.set_index(['ObjectId', 'Time'], inplace=True)

    def _get_short_lived_dead_objects(self):
        """Returns id of objects that are currently dead and have short lived"""
        object_ids = []
        for this_id, ikf in self._history_kf.items():
            if this_id not in self._past_objects:
                if ikf.lifespan_to_last_update < self.olm.min_lifespan:
                    object_ids.append(this_id)
        return object_ids

    def _get_short_lived_objects_upto_current_time(self):
        """Returns id of objects shortlived upto the current time"""
        object_ids = []
        for this_id, this_kf in self._history_kf.items():
            span_time = self._cur_time - this_kf.start_time
            if span_time < self.olm.min_lifespan:
                object_ids.append(this_id)
        return object_ids

    def _update_past_objects(self):
        """Update the list of objects that has past their prediction"""
        for this_id, ikf in self._history_kf.items():
            if self._cur_time - ikf.last_update_at > self.olm.pred_lifespan:
                self._
                ikf.forecast_upto(ikf.last_update_at + self.olm.pred_lifespan)
                if ikf.lifespan_to_last_update < self.olm.min_lifespan:
                    object_ids.append(this_id)

    def _forecast_active_objects(self):
        """Returns loglik of currently tracked objects"""
        self._logliks.clear()
        # get loklik of exisiting vehicles
        for this_id, this_kf in self._history_kf.items():
            if self._cur_time - this_kf.last_update_at < self.olm.pred_lifespan:
                this_kf.forecast_upto(self._cur_time)
                this_kf.H = self._cur_om.H.copy()
                this_kf.R = self._cur_om.R.copy()
                self._logliks[this_id] = this_kf.get_loglik_of_obs(
                    y_obs=self._cur_obs,
                    ignore_obs_inds=self._cur_om.ignore_inds_for_loglik
                )

    def _print_fusion_step_info(self):
        """Prints info of the currently tracked objects"""
        print(f'___t={self._cur_time} from {self._cur_sensor}___')
        print('Logliks:', [np.around(v, 2)for k, v in self._logliks.items()])
        print('Updates:', [np.around(v.last_update_at, 3)
                           for k, v in self._history_kf.items()])
        print('lifespan:', [np.around(v.lifespan_to_last_update, 3)
                            for k, v in self._history_kf.items()])

    def __str__(self):
        """Prints basic info"""
        out_str = f'----{self.name}----\n'
        out_str += f'KF Base object  : {self.kf_base.__class__.__name__}\n'
        istr = ', '.join([f'{k}={v}' for k, v in self.olm.__dict__.items()])
        out_str += f'Lifespan Manager: {istr}\n'
        out_str += f'Objects with life <{self.olm.min_lifespan}s are killed\n'
        out_str += f'Predicting for {self.olm.pred_lifespan}s before killing\n'

        return out_str

    def run(
        self,
        object_list: pd.DataFrame,
        obs_models: dict[str, ObservationModel],
        # list_of_sensors: List[TrafficSensor],
        data_colname: str = 'Data',
        sensor_colname: str = 'SensorID',
        etime_colname: str = 'TimeElapsed'
    ):
        """Run the fusion engine"""
        if self.verbose:
            print(f'----{self.name}----')
            print(
                f'{object_list.shape[0]} detections from {len(obs_models)} sensors')
        start_time = time.time()
        self._history_kf.clear()
        object_id = 0
        for _, irow in object_list.iterrows():
            # get next observation info
            self._cur_sensor = irow[sensor_colname]
            self._cur_time = np.around(irow[etime_colname], 3)
            self._cur_obs = np.asarray(irow[data_colname])
            self._cur_om = obs_models[self._cur_sensor]

            # update active objects and kill old dead objects
            self._forecast_active_objects()
            dead_object_ids = self._get_short_lived_dead_objects()
            self._kill_these_objects(dead_object_ids)
            if self.verbose:
                self._print_fusion_step_info()

            # data association
            probable_objects = {}
            for k, v in self._logliks.items():
                if v >= self.olm.loglik_threshold:
                    probable_objects[k] = v
            if bool(probable_objects):
                found_this_object = max(self._logliks, key=self._logliks.get)
                self._history_kf[found_this_object].y = self._cur_obs
                self._history_kf[found_this_object].update()
            else:
                start_mean = self._cur_om.H.T @ self._cur_obs
                self._initiate_new_object(object_id, start_mean)
                object_id += 1

        # dead_object_ids = self._get_short_lived_objects_upto_current_time()
        # self._kill_these_objects(dead_object_ids)
        self._assemble_df()

        # onject
        if self.verbose:
            print(f'Got a total of {len(self._history_kf)} object(s)')
            run_time = np.around(((time.time() - start_time)) / 60., 2)
            print(f'---took {run_time} mins\n', flush=True)

    @ property
    def df(self):
        """Returns fused object list"""
        return self._df

    @ property
    def history_kf(self):
        """Returns fused object list"""
        return self._history_kf
