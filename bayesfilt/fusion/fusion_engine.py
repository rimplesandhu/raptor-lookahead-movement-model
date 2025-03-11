#!/usr/bin/env python
"""Multisensor fusion engine class"""

# pylint: disable=invalid-name
import sys
import pickle
from copy import deepcopy
from typing import List, Dict, Callable
from itertools import count
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from numpy import ndarray
from tqdm import tqdm
from bayesfilt.filters import KalmanFilterBase
from bayesfilt.models import LinearObservationModel
from .utils import run_loop


@dataclass(frozen=True)
class FusionSensor:
    """Sensor class for fusion"""
    name: str = field(init=True, repr=True)
    om:LinearObservationModel=field(init=True, repr=True)
    start_mean_func: Callable = field(init=True, repr=False)
    obs_covariance: ndarray = field(init=True, repr=False)
    #get_yinds_to_ignore: callable = None
    ignore_yinds_for_association: ndarray | None = None


class FusionEngine:
    """Mult sensor fusion setup"""

    def __init__(
        self,
        kf_base: KalmanFilterBase,
        gate_threshold: float,
        pred_lifespan: float,
        name: str = 'NREL IPC Fusion Engine',
        verbose: bool = False
    ):
        # time invariant parameters
        self.name: str = name
        self.kf_base: KalmanFilterBase = kf_base
        self.gate_threshold: float = gate_threshold
        self.pred_lifespan: float = pred_lifespan
        self.sensors: Dict[str, FusionSensor] = {}
        self.verbose: bool = verbose

        # for each fusion step
        self.active_objects: List[int] = []
        self.ignored_objects: List[int] = []
        self._iter_id: count = count(start=0, step=1)
        self._cur_sensor: FusionSensor = None
        self._prev_sensor_name: str = ''
        self._cur_time: float = None
        self._prev_time: float = -np.inf

        # for logging
        self._raw: Dict[int, KalmanFilterBase] = {}

    def __repr__(self) -> str:
        """repr"""
        cls = self.__class__.__name__
        istr = f'{cls}(\n  name={self.name},\n  Filter={self.kf_base},'
        snames = ','.join([iy.name for ix,iy in self.sensors.items()])
        istr += f'\n  Sensors=[{snames}]'
        istr += f'\n  Gate_threshold={self.gate_threshold},'
        istr += f'\n  Pred_lifespan={self.pred_lifespan},'
        istr += '\n)'
        return istr

    @property
    def objects(self) -> dict:
        """returns dataframe"""
        return self._raw

    def pickle_save(self, fpath: str):
        """Save the objects as pickle file"""
        with open(fpath, 'wb') as outp:
            pickle.dump(self.objects, outp, pickle.HIGHEST_PROTOCOL)

    def _initiate_new_object(self, at_id: int, at_mean: ndarray):
        """Initiates new KF object"""
        # create new KalmanFilter object using the base object
        self._raw[at_id] = deepcopy(self.kf_base)
        self._raw[at_id].object_id = at_id  # object id
        self._raw[at_id].initiate(  # initiate the state mean for the new object
            t0=self._cur_time,
            m0=at_mean,
            flag=self._cur_sensor.name
        )
        self.active_objects.append(deepcopy(at_id))

    def smoother(self, ncores: int = 1):
        """Smooth the fused object list"""
        def run_func(pair: KalmanFilterBase):
            k, v = pair
            v.smoother(disable_pbar=True)
            return (k, v)

        results = run_loop(
            func=run_func,
            tqdm_pbar=tqdm(
                iterable=[(k, v) for k, v in self._raw.items()],
                total=len(self._raw),
                desc='Fusion-Smoother',
                leave=True,
                ncols=80,
                file=sys.stdout,
                disable=False
            ),
            ncores=ncores
        )
        self._raw = {k: v for k, v in results}

    def get_df(self, **kwargs):
        """Assemble the df with fused object list"""
        # assemble all the dfs from KalmanFilterBase Object
        tloop = tqdm(
            iterable=list(self._raw.keys()),
            total=len(self._raw),
            desc='Fusion-Gather',
            leave=True,
            ncols=80,
            file=sys.stdout,
            disable=False
        )
        list_of_dfs = []
        for ix in tloop:
            idf = self._raw[ix].get_df(**kwargs)
            idf['ObjectId'] = self._raw[ix].object_id
            list_of_dfs.append(idf)

        if list_of_dfs:
            _df = pd.concat(list_of_dfs, ignore_index=True, axis=0)
            _df.set_index(
                keys=['ObjectId', 'TimeElapsed'],
                inplace=True,
                drop=False
            )
            _df.sort_index(inplace=True)

        # change float64 to float32 for large dataframes
        if _df.memory_usage(index=True).sum()/1024/1024 > 10.:
            f64_cols = list(_df.select_dtypes(include='float64'))
            _df[f64_cols] = _df[f64_cols].astype('float32')
            _df['Flag'] = _df['Flag'].astype('category')
            # self._df['TimeElapsed'] = self._df['TimeElapsed'].round(3)
        return _df

    def _forecast_active_objects(self):
        """Returns loglik of active objects"""
        dead_objects = []
        for this_id in self.active_objects:
            this_kf = self._raw[this_id]
            pred_life = self._cur_time - this_kf.vars.t_last_update
            if pred_life <= self.pred_lifespan:
                this_kf.forecast_upto(
                    upto_time=self._cur_time,
                    flag='Forecast'
                )
            else:
                dead_objects.append(this_id)

        # stop tracking objects with predicted lifespan more than specified
        self.active_objects = [
            ix for ix in self.active_objects if ix not in dead_objects]

    def __str__(self):
        """Prints basic info"""
        out_str = f'----{self.name}----\n'
        out_str += f'KF Base object  : {self.kf_base.__class__.__name__}\n'
        # istr = ', '.join([f'{k}={v}' for k, v in self.olm.__dict__.items()])
        return out_str

    def add_sensor(self, *args, **kwargs) -> None:
        """Add sensor to fusion engine"""
        isensor = FusionSensor(*args, **kwargs)
        self.sensors[isensor.name] = isensor

    def restart(self) -> None:
        """Restart the fusion simulatio"""
        self._raw.clear()
        self._iter_id = count(start=0, step=1)
        self.active_objects = []
        self.history_da = []

    def _run_step(
            self,
            itime: float,
            isensor: str,
            idata: ndarray,

    ):
        """Fusion step"""
        # update current sensor and time
        self._cur_sensor = self.sensors[isensor]
        self._cur_time = itime
        if self.verbose:
            print(f'\n___t={self._cur_time} from {self._cur_sensor.name}')

        # forecast currently active objects
        if self._cur_time > self._prev_time:
            self._forecast_active_objects()
        if self.verbose:
            istr = [str(ix) for ix in self.active_objects]
            print('Active:  ' + ','.join(istr))

        # check if same sensor and same time
        # if yes then ignore already updated objects for the next data
        if not (
            (self._prev_time == self._cur_time) and
            (self._prev_sensor_name == self._cur_sensor.name)
        ):
            self.ignored_objects.clear()
        else:
            if self.verbose:
                istr = [str(ix) for ix in self.ignored_objects]
                print('Ignored: ' + ','.join(istr))

        # get objects that could be considered associated to this data
        likely_objects = []
        for this_id in self.active_objects:
            if this_id not in self.ignored_objects:
                this_kf = self._raw[this_id]
                # ignore_yinds = self._cur_sensor.get_yinds_to_ignore(
                #     this_kf.vars.m)
                ignore_yinds = self._cur_sensor.ignore_yinds_for_association
                this_kf.mat_H = self._cur_sensor.om.Hmat
                this_kf.update(
                    obs_y=np.asarray(idata),
                    obs_R=self._cur_sensor.obs_covariance,
                    dummy=True,
                    ignore_obs_inds=ignore_yinds
                )
                likely_objects.append(this_id)

        # find objects where obs falls within the validation region
        object_probs = {}
        for this_id in likely_objects:
            this_kf = self._raw[this_id]
            # Calculate prob directly rather than relying
            # on gate threshold which may change with observation variables
            # one less parameter to worry about
            if this_kf.metrics.NIS < self.gate_threshold:
                object_probs[this_id] = deepcopy(this_kf.metrics.LogLik)
        if self.verbose and likely_objects:
            idict = {k: self._raw[k].metrics.NIS for k in likely_objects}
            istr = [f'{k}:{np.around(v, 2)}' for k, v in idict.items()]
            print('NIS:     ' + ', '.join(istr))

        # find the object associated to the obs
        if bool(object_probs):
            # compute probs
            object_probs = {ix: np.exp(iy) for ix, iy in object_probs.items()}
            sum_ll = sum([ix for _, ix in object_probs.items()])
            object_probs = {ix: iy/sum_ll for ix, iy in object_probs.items()}

            # select using probs
            detected_object = max(object_probs, key=object_probs.get)
            self._raw[detected_object].mat_H = self._cur_sensor.om.Hmat
            self._raw[detected_object].update(
                obs_y=np.asarray(idata),
                obs_R=self._cur_sensor.obs_covariance,
                dummy=False,
                flag=self._cur_sensor.name
            )
            if self.verbose:
                istr = [f'{k}:{np.around(v, 2)}' for k,
                        v in object_probs.items()]
                print('Probs:   ' + ', '.join(istr))

        else:
            detected_object = next(self._iter_id)
            self._initiate_new_object(
                at_id=detected_object,
                at_mean=self._cur_sensor.start_mean_func(idata)
            )
        if self.verbose:
            print(f'Selected:{detected_object}')

        # finish
        #self.ignored_objects.append(deepcopy(detected_object))
        self._prev_time = deepcopy(self._cur_time)
        self._prev_sensor_name = deepcopy(self._cur_sensor.name)

    def run(
        self,
        list_of_time: list[float],
        list_of_data: list[np.ndarray],
        list_of_sensors: list[str],
        restart: bool = True
    ):
        """Run the fusion engine"""

        # check compatibility of data
        ndata = len(list_of_data)
        assert ndata == len(list_of_time), 'datatime length mismatch'
        assert ndata == len(list_of_sensors), 'sensors/data length mismatch'

        # make sure sensors found in data are defined
        data_sensors = np.unique(list_of_sensors)
        fusion_sensors = list(self.sensors.keys())
        for iname in data_sensors:
            assert iname in fusion_sensors, f'{iname} sensor not defined!'

        istr = ','.join(data_sensors)
        print(f'Fusion: Fusing {ndata} detections from sensors: {istr}')
        # start_time = time.time()
        if restart:
            self.restart()

        loop = tqdm(
            iterable=range(ndata),
            total=ndata,
            position=0,
            leave=True,
            ncols=80,
            file=sys.stdout,
            desc='Fusion-Run'
        )
        for ith in loop:
            self._run_step(
                itime=list_of_time[ith],
                isensor=list_of_sensors[ith],
                idata=np.asarray(list_of_data[ith])
            )

        # finish
        # print(f'Fusion: Fused object list contains {len(self._raw)} object(s)')
        # run_time = np.around(((time.time() - start_time)) / 60., 2)
        # print(f'Fusion: took {run_time} mins', flush=True)

    # def _get_short_lived_objects_upto_current_time(self):
    #     """Returns id of objects shortlived upto the current time"""
    #     object_ids = []
    #     for this_id, this_kf in self._raw.items():
    #         span_time = self._cur_time - this_kf.start_time
    #         if span_time < self.olm.min_lifespan:
    #             object_ids.append(this_id)
    #     return object_ids

    # def _update_past_objects(self):
    #     """Update the list of objects that has past their prediction"""
    #     object_ids = []
    #     for this_id, ikf in self._raw.items():
    #         if self._cur_time - ikf.last_update_at > self.olm.pred_lifespan:
    #             self._
    #             ikf.forecast_upto(ikf.last_update_at + self.olm.pred_lifespan)
    #             if ikf.lifespan_to_last_update < self.olm.min_lifespan:
    #                 object_ids.append(this_id)
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
    #     self._raw.clear()
    #     object_id = 0
    #     self.active_objects = []
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
    #             if v >= self.olm.gate_threshold:
    #                 probable_objects[k] = v
    #         if bool(probable_objects):
    #             found_this_object = max(self._logliks, key=self._logliks.get)
    #             self._raw[found_this_object].y = self._cur_obs
    #             self._raw[found_this_object].update()
    #         else:
    #             start_mean = self._cur_om.H.T @ self._cur_obs
    #             self._initiate_new_object(object_id, start_mean)
    #             object_id += 1

    #     # dead_object_ids = self._get_short_lived_objects_upto_current_time()
    #     # self._kill_these_objects(dead_object_ids)
    #     self._assemble_df()

    #     # finish
    #     print(f'Got a total of {len(self._raw)} object(s)')
    #     run_time = np.around(((time.time() - start_time)) / 60., 2)
    #     print(f'---took {run_time} mins\n', flush=True)

    # def _remove_short_lived_inactive_objects(self):
    #     """Returns id of objects that are currently dead and have short lived"""
    #     for this_id in list(self._raw.keys()):
    #         if this_id not in self.active_objects:
    #             ikf = self._raw[this_id]
    #             lifespan = ikf.vars.t_last_update - ikf.vars.t_start
    #             if lifespan < self.olm.min_lifespan:
    #                 if self.verbose:
    #                     print(f'Deleting {this_id}, lived for {lifespan}s')
    #                 del self._raw[this_id]

    # def _kill_these_objects(self, these_ids: List[int]):
    #     """Kill/rmeove this object"""
    #     for this_id in list(these_ids):
    #         if self.verbose:
    #             istr = np.around(self._cur_time, 3)
    #             print(f'Killed {this_id} at {istr} sec')
    #         del self._raw[this_id]
        # new_entry = dict(
        #     ObjectId=object_id,
        #     DaLogLik=np.nan,
        #     TimeElapsed=self._cur_time,
        #     Sensor=self._cur_sensor
        # )
        # self.history_da.append(new_entry)
