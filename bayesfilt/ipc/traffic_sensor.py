#!/usr/bin/env python
"""Traffic sensor class"""

# pylint: disable=invalid-name
import datetime
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import patches
from bayesfilt.observation_model import ObservationModel
from bayesfilt.kalman_filter_base import KalmanFilterBase


class TrafficSensor:
    """Base class for defining a traffic sensor that produces an object list"""

    def __init__(
        self,
        sensor_name: str,
        data_df: pd.DataFrame | Dict,
        observation_model: ObservationModel | None = None,
        time_minmax: Tuple[datetime.time, datetime.time] | None = None
    ):

        self.name = sensor_name
        self.om = observation_model
        self.time_col = 'Time'
        self.x_col = 'X'
        self.y_col = 'Y'
        self.width_col = 'Width'
        self.length_col = 'Length'
        self.heading_col = 'Heading'
        self.quality_col = 'TrackQuality'
        self.id_col = 'ObjectID'

        self._df = pd.DataFrame(data_df)
        self._df.set_index(pd.to_datetime(self._df[self.time_col]),
                           inplace=True, drop=True)
        if time_minmax is not None:
            self._df = self.df.between_time(*time_minmax)
        self.states = list(self.df.columns)
        assert self.x_col in self.states, f'Need {self.x_col} in data_df!'
        assert self.y_col in self.states, f'Need {self.y_col} in data_df!'
        #self._df.set_index(pd.to_datetime(time_series), inplace=True)
        self._df.sort_index(inplace=True)
        float64_cols = list(self.df.select_dtypes(include='float64'))
        self._df[float64_cols] = self.df[float64_cols].astype('float32')
        int64_cols = list(self.df.select_dtypes(include='int64'))
        self._df[int64_cols] = self.df[int64_cols].astype('int32')

        if self.heading_col in self.states:
            self.object_shape = 'rectangle'
            assert self.width_col in self.df.columns, 'Missing width info!'
            assert self.length_col in self.df.columns, 'Missing length info!'
            self.boxstyle = patches.BoxStyle('round', rounding_size=1.)
        else:
            self.object_shape = 'circle'
            self.boxstyle = patches.BoxStyle('circle', pad=0.3)

    @ property
    def df(self) -> pd.DataFrame:
        """Returns dataframe containing all the data"""
        return self._df

    def plot_path(self, ax, *args, **kwargs) -> None:
        """plot the path on xy plane this run"""
        ax.plot(self.df.loc[:, self.x_col],
                self.df.loc[:, self.y_col],
                *args, **kwargs)

    def draw_namebox(self, ax, xyloc: Tuple[float, float], **kwargs) -> None:
        """Draw a box with name of the sensor"""
        ax.text(
            *xyloc,
            self.name,
            transform=ax.transAxes,
            fontsize=10,
            **kwargs
        )

    def draw_frame(
        self,
        ax,
        at_time: datetime.time,
        time_padding: float,
        show_id: bool = False,
        use_alpha: bool = False,
        **kwargs
    ) -> None:
        """Draws frame at this time"""
        time_low = at_time - datetime.timedelta(seconds=time_padding)
        time_upp = at_time + datetime.timedelta(seconds=time_padding)
        dfshort = self.df.between_time(time_low.time(), time_upp.time())
        # print(dfshort.shape[0])
        for _, irow in dfshort.iterrows():
            centroid = (irow[self.x_col], irow[self.y_col])
            lowerleft_loc = (centroid[0], centroid[1])
            width, length = (1.75, 1.75)
            rot_transform = ax.transData
            if self.object_shape == 'rectangle':
                width = irow[self.width_col]
                length = irow[self.length_col]
                lowerleft_loc = (
                    centroid[0] - width / 2.,
                    centroid[1] - length / 2.
                )
                angle = (irow[self.heading_col] + np.pi / 2) % (2.0 * np.pi)
                rot_transform = mpl.transforms.Affine2D().rotate_deg_around(
                    centroid[0],
                    centroid[1],
                    np.degrees(angle)
                ) + ax.transData
            alpha = 0.8
            if self.quality_col in self.df.columns:
                if use_alpha:
                    alpha = irow[self.quality_col]
            ibox = patches.FancyBboxPatch(
                xy=lowerleft_loc,
                width=width,
                height=length,
                boxstyle=self.boxstyle,
                transform=rot_transform,
                alpha=alpha,
                **kwargs
            )
            ax.add_patch(ibox)
            if self.id_col in self.df.columns:
                if show_id:
                    ax.text(
                        irow[self.x_col],
                        irow[self.y_col],
                        irow[self.id_col],
                        fontsize=9,
                        ha='center',
                        va='center',
                        color='w'
                    )

    def get_object_list(self, states: List[str]) -> pd.DataFrame:
        """Returns object list"""
        for iname in states:
            out_str = f'{iname} not found! Choose among {self.states}'
            assert iname in self.states, out_str
        olist_data = self.df.loc[:, states].values.tolist()
        olist_index = self.df.index
        out_df = pd.DataFrame({'ObjectList': olist_data}, index=olist_index)
        return out_df

    def __str__(self) -> str:
        out_str = f'----{self.name}----\n'
        test_date = self.df.index.max().date().strftime('%F')
        out_str += f'Date of test : {test_date}\n'
        start_time = self.df.index.min().time().strftime('%H:%M:%S')
        end_time = self.df.index.max().time().strftime('%H:%M:%S')
        out_str += f'Time range   : {start_time}-{end_time}\n'
        out_str += f'# of entries : {self.df.shape[0]}\n'
        return out_str


def merge_traffic_sensor_data(
    list_of_sensors: List[TrafficSensor]
) -> pd.DataFrame:
    """Merge data from two sensors"""
    list_of_dfs = []
    for ith_sensor in list_of_sensors:
        ith_df = ith_sensor.get_object_list(ith_sensor.om.obs_names)
        ith_df['SensorID'] = [ith_sensor.name] * ith_df.shape[0]
        list_of_dfs.append(ith_df)
    sdf = pd.concat(list_of_dfs)
    sdf['SensorID'] = sdf['SensorID'].astype('category')
    sdf.sort_index(inplace=True)
    sdf['TimeElapsed'] = sdf.index.to_series().diff().dt.total_seconds()
    sdf['TimeElapsed'] = sdf['TimeElapsed'].fillna(0.).cumsum()
    return sdf
