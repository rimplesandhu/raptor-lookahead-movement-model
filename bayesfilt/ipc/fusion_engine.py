#!/usr/bin/env python
"""Multisensor fusion engine class"""

# pylint: disable=invalid-name
import sys
import time
from copy import deepcopy
from typing import List
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm
from bayesfilt.filters import KalmanFilterBase
from bayesfilt.models import ObservationModel


@dataclass
class ObjectLifespanManager:
    """Lifespan Management settings"""
    loglik_threshold: float
    pred_lifespan: float
    min_lifespan: float
    # vehicle_lanes: MultiPolygon | None = None

    def __post_init__(self):
        """Post initiation"""
        print('----Lifespan Manager----')
        print(f'Lifespan=Duration from first update till last update')
        print(f'Rules: Lifespan<{self.min_lifespan}s -> Remove from log!')
        print(f'Rules: No detection for {self.pred_lifespan}s -> Inactive!')
        print('')


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
        self.active_object_ids = []
        self.history_kf = {}
        self.history_da = []
        self._cur_sensor = None
        self._cur_time = None
        self._cur_obs = None
        self._cur_om = None
        self._df_kf = None
        self._df_da = None
        self._logliks = {}

    @property
    def df(self) -> pd.DataFrame:
        """returns dataframe"""
        return self._df_kf.copy()

    def _initiate_new_object(self, at_id, at_mean):
        """Initiates new object id"""
        if self.verbose:
            print(f'Started {at_id} at {np.around(self._cur_time, 3)}s')
        self.history_kf[at_id] = deepcopy(self.kf_base)
        self.history_kf[at_id].objectid = at_id
        self.history_kf[at_id].initiate(t0=self._cur_time, m0=at_mean)
        self.active_object_ids.append(at_id)

    def _kill_these_objects(self, these_ids: List[int]):
        """Kill/rmeove this object"""
        for this_id in list(these_ids):
            if self.verbose:
                print(
                    f'Killed {this_id} at {np.around(self._cur_time, 3)}s')
            del self.history_kf[this_id]

    def assemble_dataframe(self, **kwargs):
        """Assemble the df with fused object list"""
        # assemble all the dfs from KalmanFilterBase Object
        list_of_dfs = []
        for ix in list(self.history_kf.keys()):
            list_of_dfs.append(self.history_kf[ix].get_df(**kwargs))

        if list_of_dfs:
            self._df_kf = pd.concat(list_of_dfs, ignore_index=True, axis=0)
        else:
            self._df_kf = pd.DataFrame({})
        # self._df_kf.drop(columns=['Observation', 'Training', 'FilterNEES',
        #                        'FilterNIS'], inplace=True, errors='ignore')
        # self._df_kf['Time'] = [start_datetime + datetime.timedelta(
        #     milliseconds=1000 * ix) for ix in self._df_kf['TimeElapsed']]
        # self._df_kf.set_index(['ObjectId', 'Time'], inplace=True)

    def _remove_short_lived_inactive_objects(self):
        """Returns id of objects that are currently dead and have short lived"""
        for this_id in list(self.history_kf.keys()):
            if this_id not in self.active_object_ids:
                this_kf = self.history_kf[this_id]
                if this_kf.lifespan_to_last_update < self.olm.min_lifespan:
                    if self.verbose:
                        print(f'Deleting {this_id}, only lived for {
                            this_kf.lifespan_to_last_update}s')
                    del self.history_kf[this_id]

    # def _get_short_lived_objects_upto_current_time(self):
    #     """Returns id of objects shortlived upto the current time"""
    #     object_ids = []
    #     for this_id, this_kf in self.history_kf.items():
    #         span_time = self._cur_time - this_kf.start_time
    #         if span_time < self.olm.min_lifespan:
    #             object_ids.append(this_id)
    #     return object_ids

    # def _update_past_objects(self):
    #     """Update the list of objects that has past their prediction"""
    #     object_ids = []
    #     for this_id, ikf in self.history_kf.items():
    #         if self._cur_time - ikf.last_update_at > self.olm.pred_lifespan:
    #             self._
    #             ikf.forecast_upto(ikf.last_update_at + self.olm.pred_lifespan)
    #             if ikf.lifespan_to_last_update < self.olm.min_lifespan:
    #                 object_ids.append(this_id)

    def _forecast_active_objects(self):
        """Returns loglik of active objects"""
        self._logliks.clear()
        dead_objects = []
        for this_id in self.active_object_ids:
            this_kf = self.history_kf[this_id]
            if self._cur_time - this_kf.last_update_at <= self.olm.pred_lifespan:
                this_kf.forecast_upto(self._cur_time)
                this_kf.H = self._cur_om.H.copy()
                this_kf.R = self._cur_om.R.copy()
                self._logliks[this_id] = this_kf.get_loglik_of_obs(
                    y_obs=self._cur_obs,
                    ignore_obs_inds=self._cur_om.ignore_inds_for_loglik
                )
            else:
                dead_objects.append(this_id)
        self.active_object_ids = [
            ix for ix in self.active_object_ids if ix not in dead_objects]

    def _print_fusion_step_info(self):
        """Prints info of the currently tracked objects"""
        print(f'___t={self._cur_time} from {self._cur_sensor}___')
        print('Logliks:', [np.around(v, 2)for k, v in self._logliks.items()])
        print('Updates:', [np.around(v.last_update_at, 3)
                           for k, v in self.history_kf.items()])
        print('lifespan:', [np.around(v.lifespan_to_last_update, 3)
                            for k, v in self.history_kf.items()])

    def __str__(self):
        """Prints basic info"""
        out_str = f'----{self.name}----\n'
        out_str += f'KF Base object  : {self.kf_base.__class__.__name__}\n'
        istr = ', '.join([f'{k}={v}' for k, v in self.olm.__dict__.items()])

        return out_str

    def run(
        self,
        dict_of_obs_models: dict[str, ObservationModel],
        list_of_time: list[float],
        list_of_detections: list[np.ndarray],
        list_of_sensors: list[str],
        restart: bool = True
    ):
        """Run the fusion engine"""

        # check compatibility of data
        assert len(list_of_detections) == len(list_of_time), 'Shape mismatch!'
        unique_sensors = np.unique(list_of_sensors)

        print(f'----{self.name}----')
        print(f'{len(list_of_time)} detections from {
              len(unique_sensors)} sensor(s)')
        start_time = time.time()
        if restart:
            self.history_kf.clear()
            object_id = 0
            self.active_object_ids = []
            self.history_da = []
        else:
            object_id = max(list(self.history_kf.keys())) + 1

        loop = tqdm(
            iterable=range(len(list_of_detections)),
            total=len(list_of_detections),
            position=0,
            leave=True,
            file=sys.stdout,
            desc='Fusion'
        )
        for ith in loop:
            # get next observation info
            self._cur_sensor = list_of_sensors[ith]
            self._cur_time = list_of_time[ith]
            self._cur_obs = np.asarray(list_of_detections[ith])
            self._cur_om = dict_of_obs_models[self._cur_sensor]

            # forecast currently active objects
            self._forecast_active_objects()
            # self._remove_short_lived_inactive_objects()

            # print info
            if self.verbose:
                self._print_fusion_step_info()

            # data association
            probable_objects = {}
            for k, v in self._logliks.items():
                if v >= self.olm.loglik_threshold:
                    probable_objects[k] = v
            if bool(probable_objects):
                found_this_object = max(
                    self._logliks, key=self._logliks.get)
                self.history_kf[found_this_object].y = self._cur_obs
                self.history_kf[found_this_object].update()
                new_entry = dict(
                    ObjectId=found_this_object,
                    DaLogLik=self._logliks[found_this_object],
                    TimeElapsed=self._cur_time,
                    Sensor=self._cur_sensor
                )
                self.history_da.append(new_entry)
            else:
                start_mean = self._cur_om.H.T @ self._cur_obs
                self._initiate_new_object(object_id, start_mean)
                new_entry = dict(
                    ObjectId=object_id,
                    DaLogLik=np.nan,
                    TimeElapsed=self._cur_time,
                    Sensor=self._cur_sensor
                )
                self.history_da.append(new_entry)
                object_id += 1

        # dead_object_ids = self._get_short_lived_objects_upto_current_time()
        # self._kill_these_objects(dead_object_ids)
        self._df_da = pd.DataFrame(self.history_da)

        # finish
        print(f'Got a total of {len(self.history_kf)} object(s)')
        run_time = np.around(((time.time() - start_time)) / 60., 2)
        print(f'---took {run_time} mins\n', flush=True)

    # def run(
    #     self,
    #     object_list: pd.DataFrame,
    #     obs_models: dict[str, ObservationModel],
    #     # list_of_sensors: List[TrafficSensor],
    #     data_colname: str = 'Data',
    #     sensor_colname: str = 'Sensor',
    #     etime_colname: str = 'TimeElapsed'
    # ):
    #     """Run the fusion engine"""
    #     print(f'----{self.name}----')
    #     print(f'{object_list.shape[0]} detections from {
    #           len(obs_models)} sensor(s)')
    #     start_time = time.time()
    #     self.history_kf.clear()
    #     object_id = 0
    #     self.active_object_ids = []
    #     loop = tqdm(
    #         iterable=object_list.iterrows(),
    #         total=len(object_list),
    #         position=0,
    #         leave=True,
    #         file=sys.stdout,
    #         desc='Fusion'
    #     )
    #     for _, irow in loop:
    #         # get next observation info
    #         self._cur_sensor = irow[sensor_colname]
    #         self._cur_time = np.around(irow[etime_colname], 3)
    #         self._cur_obs = np.asarray(irow[data_colname])
    #         self._cur_om = obs_models[self._cur_sensor]

    #         # forecast currently active objects

    #         self._forecast_active_objects()
    #         self._remove_short_lived_inactive_objects()

    #         # dead_object_ids = self._get_short_lived_dead_objects()
    #         # self._kill_these_objects(dead_object_ids)
    #         if self.verbose:
    #             self._print_fusion_step_info()

    #         # data association
    #         probable_objects = {}
    #         for k, v in self._logliks.items():
    #             if v >= self.olm.loglik_threshold:
    #                 probable_objects[k] = v
    #         if bool(probable_objects):
    #             found_this_object = max(self._logliks, key=self._logliks.get)
    #             self.history_kf[found_this_object].y = self._cur_obs
    #             self.history_kf[found_this_object].update()
    #         else:
    #             start_mean = self._cur_om.H.T @ self._cur_obs
    #             self._initiate_new_object(object_id, start_mean)
    #             object_id += 1

    #     # dead_object_ids = self._get_short_lived_objects_upto_current_time()
    #     # self._kill_these_objects(dead_object_ids)
    #     self._assemble_df()

    #     # finish
    #     print(f'Got a total of {len(self.history_kf)} object(s)')
    #     run_time = np.around(((time.time() - start_time)) / 60., 2)
    #     print(f'---took {run_time} mins\n', flush=True)
