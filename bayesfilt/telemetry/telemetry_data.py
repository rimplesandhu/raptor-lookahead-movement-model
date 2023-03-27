""" Base class for telemetry data """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
from typing import Callable
from functools import partial
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy import ndarray
from .utils import get_bin_edges


@dataclass
class TelemetryData:
    """Telemetry data class"""
    df: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    geo_crs = 'EPSG:4326'
    proj_crs: str = 'ESRI:102008'
    x_col: str = 'PositionX'
    y_col: str = 'PositionY'
    z_col: str = 'Altitude'
    t_col: str = 'TimeUTC'
    id_col: str = 'AnimalID'
    region_col: str = 'Group'
    domain_col = 'DomainID'
    tdiff_col = 'TimeDiff'
    xvel_col = 'VelocityX'
    yvel_col = 'VelocityY'
    heading_col = 'Heading'
    hvel_col = 'VelocityHor'
    zvel_col = 'VelocityVer'
    falsefix_col = 'FalseFix'
    trackid_col = 'TrackID'
    tracktime_col = 'TrackTimeElapsed'
    df_subdomains: pd.DataFrame = field(
        default_factory=pd.DataFrame, repr=False)
    log_level: int = 40

    def __post_init__(self):
        """Post init function"""
        # set up logger
        logging.basicConfig(
            level=self.log_level,
            format='%(name)-12s:%(levelname)-s: %(message)s',
        )
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(self.log_level)
        self.proj_crs = ccrs.CRS(self.proj_crs)
        self.geo_crs = ccrs.CRS(self.geo_crs)

        # finite diff movement variables
        self.check_validity_of_columns([self.t_col, self.x_col, self.y_col])
        tdiff = self.df[self.t_col].diff().dt.total_seconds().bfill().round(5)
        self.df[self.tdiff_col] = tdiff.astype('float32')
        xvel = self.df[self.x_col].diff().bfill().divide(tdiff)
        self.df[self.xvel_col] = xvel.astype('float32')
        yvel = self.df[self.y_col].diff().bfill().divide(tdiff)
        self.df[self.yvel_col] = yvel.astype('float32')
        hvel = np.sqrt(xvel**2 + yvel**2)
        self.df[self.hvel_col] = hvel.astype('float32')
        heading = (np.degrees(np.arctan2(xvel, yvel)))
        self.df[self.heading_col] = heading.astype('float32')
        if self.z_col in self.df.columns:
            zvel = self.df[self.z_col].diff().bfill().divide(tdiff)
            self.df[self.zvel_col] = zvel.astype('float32')

        # post process dataframe, change dtypes
        self.df[self.t_col] = pd.to_datetime(self.df[self.t_col])
        if self.region_col not in self.df.columns:
            self.df[self.region_col] = 'USA'
        self.df[self.region_col] = self.df[self.region_col].astype('category')
        if self.id_col not in self.df.columns:
            self.df[self.id_col] = 0
        self.df[self.id_col] = self.df[self.id_col].astype('category')
        self.df[self.falsefix_col] = False

        # warn about memory usage
        total_mem = np.around(self.df.memory_usage().sum() / 1024 / 1024, 2)
        if total_mem > 1024:
            self.log.warning(f'Memory usage of {total_mem} MB')
        self.df.info(verbose=True, memory_usage=True, show_counts=True)

    def sort_df(
        self,
        by: list[str] | None = None
    ) -> None:
        """Sorts dataframe according to given columns, in order, in place"""
        by = [self.region_col, self.id_col, self.t_col] if by is None else by
        istr = ', '.join(by)
        self.printit(f'Sorting based on {istr}; in that order')
        self.check_validity_of_columns(by)
        self.df.set_index(by, inplace=True, drop=False)
        self.df.sort_index(inplace=True)
        # self.df.sort_index(axis=1, inplace=True)
        self.df.reset_index(inplace=True, drop=True)

    def plot_track_in_time(
        self,
        track_id: int,
        col_name: str,
        *args,
        ax: plt.Axes | None = None,
        **kwargs
    ):
        """Plot this tracks"""
        ax = plt.gca() if ax is None else ax
        self.check_validity_of_columns(col_name)
        dftrack = self.df[self.df[self.trackid_col] == track_id]
        tlist = dftrack[self.tracktime_col]
        cb = ax.plot(tlist, dftrack[col_name], *args, **kwargs)
        ax.set_ylabel(f'{col_name}')
        ax.set_xlim([tlist.min(), tlist.max()])
        ax.set_xlabel('Time elapsed (Seconds)')
        return cb

    def plot_track_in_space(
        self,
        track_id: int,
        *args,
        ax: plt.Axes | None = None,
        **kwargs
    ):
        """Plot this tracks"""
        ax = plt.gca() if ax is None else ax
        dftrack = self.df[self.df[self.trackid_col] == track_id]
        cb = ax.plot(dftrack[self.x_col], dftrack[self.y_col], *args, **kwargs)
        ax.set_xlabel(f'{self.x_col}')
        ax.set_ylabel(f'{self.y_col}')
        return cb

    def annotate_track_info(
        self,
        min_time_interval: float,
        min_time_duration: float,
        min_num_points: int,
        time_col: str | None = None
    ) -> None:
        """Extract tracks from telemetry data"""
        time_col = self.t_col if time_col is None else time_col
        self.sort_df([self.region_col, self.id_col, time_col])
        first_of_day = (
            (self.df[time_col].dt.date != self.df[time_col].dt.date.shift()) |
            (self.df[self.id_col].astype('int32').diff() != 0)
        )
        first_of_track = (
            (self.df[self.tdiff_col] >= min_time_interval) |
            (first_of_day) |
            self.df[self.falsefix_col]
        )
        first_of_track = (
            first_of_track |
            first_of_track.shift(1) |
            first_of_track.shift(-1)
        )
        self.df[self.trackid_col] = first_of_track.cumsum()
        grouped = self.df.groupby(self.trackid_col)[self.tdiff_col]
        track_bool = (
            (grouped.transform('sum') < min_time_duration) |
            (grouped.transform('count') < min_num_points)
        )
        self.df.loc[track_bool, self.trackid_col] = 0
        track_cat = self.df[self.trackid_col].astype('category')
        self.df[self.trackid_col] = track_cat.cat.codes
        grouped = self.df.groupby(self.trackid_col)[self.tdiff_col]
        self.df[self.tracktime_col] = grouped.transform('cumsum')
        self.df.loc[self.df[self.trackid_col]
                    == 0, self.tracktime_col] = np.nan
        # print the ingo
        ntracks = self.df.groupby(self.region_col)[self.trackid_col].nunique()
        istr = ', '.join(f'{k}={v}' for k, v in ntracks.to_dict().items())
        self.printit(f'Number of tracks: {istr}')

    def partition_into_subdomains(
        self,
        x_width: float = 20 * 1000,
        y_width: float = 20 * 1000,
        pad: float = 1000.,
        by: str | None = None
    ) -> None:
        """Parition spatial data into rectangular chunks"""
        by = self.region_col if by is None else by
        x_col = self.x_col
        y_col = self.y_col
        self.check_validity_of_columns([x_col, y_col])
        self.df[self.domain_col] = -1
        self.df_subdomains = {}
        for iregion in self.df[self.region_col].unique():
            rbool = self.df[self.region_col] == iregion
            x_edges = get_bin_edges(self.df.loc[rbool, x_col], x_width, pad)
            y_edges = get_bin_edges(self.df.loc[rbool, y_col], y_width, pad)
            x_bins = np.digitize(self.df.loc[rbool, x_col], x_edges)
            y_bins = np.digitize(self.df.loc[rbool, y_col], y_edges)
            uniq, domain_idx = np.unique(
                np.vstack((x_bins, y_bins)),
                axis=1,
                return_inverse=True
            )
            domain_id = [f'{iregion}{ix}' for ix in domain_idx]
            self.df.loc[rbool, self.domain_col] = domain_id
            for idomain in range(domain_idx.max()):
                domain_id = f'{iregion}{idomain}'
                self.df_subdomains[domain_id] = [
                    x_edges[uniq[0, idomain] - 1] - pad,
                    x_edges[uniq[0, idomain]] + pad,
                    y_edges[uniq[1, idomain] - 1] - pad,
                    y_edges[uniq[1, idomain]] + pad,
                ]
            num_domains = len(x_edges) * len(y_edges)
            self.printit(f'{iregion}={uniq.shape[1]}/{num_domains} have data')
        self.df_subdomains = pd.DataFrame.from_dict(
            self.df_subdomains,
            orient='index',
            columns=['xmin', 'xmax', 'ymin', 'ymax']
        )
        geo_corners = self.geo_crs.transform_points(
            x=self.df_subdomains[['xmin', 'xmin', 'xmax', 'xmax']].values,
            y=self.df_subdomains[['ymin', 'ymax', 'ymin', 'ymax']].values,
            src_crs=self.proj_crs
        )
        self.df_subdomains['lonmin'] = np.amin(geo_corners, axis=1)[:, 0]
        self.df_subdomains['lonmax'] = np.amax(geo_corners, axis=1)[:, 0]
        self.df_subdomains['latmin'] = np.amin(geo_corners, axis=1)[:, 1]
        self.df_subdomains['latmax'] = np.amax(geo_corners, axis=1)[:, 1]
        self.df_subdomains.index.name = self.domain_col
        self.df[self.domain_col] = self.df[self.domain_col].astype('category')

    def plot_subdomain(
        self,
        idomain: str,
        *args,
        ax: plt.Axes | None = None,
        **kwargs
    ):
        """Plots the data in this subdomain"""
        if self.df_subdomains.empty:
            self.printit('No subdomain info found!')
        else:
            valid_domains = self.df_subdomains.index.tolist()
            assert idomain in valid_domains, f'{idomain} invalid!'
            xy_bounds = self.df_subdomains.loc[idomain].values
            domain_df = self.df[self.df[self.domain_col] == idomain]
            ax = plt.gca() if ax is None else ax
            domain_box = patches.Rectangle(
                xy=(xy_bounds[0], xy_bounds[2]),
                width=xy_bounds[1] - xy_bounds[0],
                height=xy_bounds[3] - xy_bounds[2],
                fill=True, alpha=0.1
            )
            ax.add_artist(domain_box)
            ax.plot(domain_df[self.x_col],
                    domain_df[self.y_col],
                    *args, **kwargs)
            ax.set_aspect('equal')
            # ax.set_xlim(*xy_bounds[:2])

    def ignore_data_based_on_vertical_speed(
        self,
        max_change: float
    ) -> None:
        """False fix points based on vertical speed"""
        vspeed = self.df[self.zvel_col]
        vchange = vspeed - np.maximum(vspeed.shift(1), vspeed.shift(-1))
        cond1 = (vspeed >= 0.) & (vchange >= max_change)
        vchange = np.minimum(vspeed.shift(1), vspeed.shift(-1)) - vspeed
        cond2 = (vspeed <= 0.) & (vchange >= max_change)
        cond_both = (cond1) | (cond2)
        self.df[self.falsefix_col] = self.df[self.falsefix_col] | cond_both
        self.printit(f'Found {cond_both.sum()} bad points - vpseed condition')

    def ignore_data_based_on_horizontal_speed(
        self,
        min_speed: float,
        min_npoints: int = 5
    ) -> None:
        """Perched points"""
        hspeed = self.df[self.hvel_col]
        shift_periods = np.arange(-min_npoints, min_npoints + 1)
        hspeed_bool = (hspeed <= min_speed)
        for ishift in shift_periods:
            hspeed_bool = hspeed_bool & (hspeed.shift(
                ishift, fill_value=0.) <= min_speed)
        cond_bool = hspeed_bool.copy()
        for ishift in shift_periods:
            cond_bool = cond_bool | hspeed_bool.shift(ishift, fill_value=False)
        self.df[self.falsefix_col] = self.df[self.falsefix_col] | cond_bool
        self.printit(f'Found {cond_bool.sum()} bad points - hspeed condition')

    def check_validity_of_columns(self, cols):
        """check if these columns exist in the dataframe"""
        for icol in np.atleast_1d(cols):
            if icol not in list(self.df.columns):
                self.raiseit(f'{icol} not found in {list(self.df.columns)}')

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.__class__.__name__}: {outstr}')

    def printit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        print(f'{self.__class__.__name__}: {outstr}', flush=True)

    def df_track(self, track_id: int):
        """Returns track df"""
        return self.df[self.df[self.trackid_col] == track_id]

    @property
    def track_ids(self):
        """return a list of tracks"""
        out_list = []
        if self.trackid_col in self.df.columns:
            out_list = list(self.df[self.trackid_col].unique())
        return out_list
    # def get_memory(
    #     self,
    #     by: str | None = None
    # ) -> None:
    #     """Prints memory usage by specific column"""
    #     by = self.region_col if by is None else by
    #     self.check_validity_of_columns(by)
    #     total_memory = 0.
    #     for ikey in self.df[by].unique():
    #         xdf = self.df[self.df[by] == ikey]
    #         mem_mb = xdf.memory_usage(index=True, deep=True).sum()
    #         total_memory += mem_mb / 1024 / 1024
    #     return total_memory
