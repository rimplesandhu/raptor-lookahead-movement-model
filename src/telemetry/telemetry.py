""" Base class for telemetry data """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
from pathlib import Path
from typing import Callable
from functools import partial
from dataclasses import dataclass, field
import pathos.multiprocessing as mp
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import rasterio
from numpy import ndarray
from ssrs.layers import calcOrographicUpdraft_original
from ssrs.raster import transform_coordinates, transform_bounds
from .utils import get_bin_edges, get_unique_times
from ._base_data import BaseGeoData
from .data_3dep import Data3DEP
from .data_hrrr import DataHRRR
from .utils import run_loop, get_wind_direction, get_bound_from_positions


@dataclass(frozen=True)
class TelemetryAttributeNames:
    """Dataclass for defining basic attributes of Telemetry class"""
    lon_col: str = field(default='Longitude', repr=False)
    lat_col: str = field(default='Latitude', repr=False)
    x_col: str = field(default='PositionX', repr=False)
    y_col: str = field(default='PositionY', repr=False)
    z_col: str = field(default='Altitude', repr=False)
    t_col: str = field(default='TimeUTC', repr=False)
    tlocal_col: str = field(default='TimeLocal', repr=False)
    id_col: str = field(default='AnimalID', repr=False)
    region_col: str = field(default='Group', repr=False)
    domain_col: str = field(default='DomainID', repr=False)
    tdiff_col: str = field(default='TimeDiff', repr=False)
    xvel_col: str = field(default='VelocityX', repr=False)
    yvel_col: str = field(default='VelocityY', repr=False)
    heading_col: str = field(default='Heading', repr=False)
    hvel_col: str = field(default='VelocityHor', repr=False)
    zvel_col: str = field(default='VelocityVer', repr=False)
    falsefix_col: str = field(default='FalseFix', repr=False)
    trackid_col: str = field(default='TrackID', repr=False)
    tracktime_col: str = field(default='TrackTimeElapsed', repr=False)


class Telemetry(TelemetryAttributeNames, BaseGeoData):
    """Class for handling telemetry data"""

    def __init__(
        self,
        times: ndarray,
        lons: ndarray,
        lats: ndarray,
        zlocs: ndarray | None = None,
        regions: ndarray | None = None,
        animalids: ndarray | None = None,
        times_local: ndarray | None = None,
        df_add: pd.DataFrame | None = None,
        **kwargs
    ):
        # initialize
        self.printit('Initiating Telemetry object..')
        BaseGeoData.__init__(self, **kwargs)
        TelemetryAttributeNames.__init__(self)
        self.threedep_dir = self.out_dir / 'data_3dep'
        Path(self.threedep_dir).mkdir(parents=True, exist_ok=True)
        self.hrrr_dir = self.out_dir / 'data_hrrr'
        Path(self.hrrr_dir).mkdir(parents=True, exist_ok=True)
        self.domain_fpath = Path(self.threedep_dir, 'subdomains')

        # create dataframe
        self.df_subdomains: pd.DataFrame = pd.DataFrame()
        self.df = pd.DataFrame({
            self.t_col: pd.to_datetime(times, unit='ms'),
            self.lat_col: np.asarray(lats),
            self.lon_col: np.asarray(lons)
        })
        if (df_add is not None) & (isinstance(df_add, pd.DataFrame)):
            for icol in df_add.columns:
                if icol not in self.df.columns:
                    self.df[icol] = df_add[icol]

        # compute x and y positions in proj ref system
        self.printit('Computing positions in projected coordinate..')
        xylocs = self.proj_crs.transform_points(
            x=np.asarray(lons),
            y=np.asarray(lats),
            src_crs=ccrs.CRS(self.geo_crs)
        )
        self.df[self.x_col] = np.asarray(xylocs[:, 0]).astype('float64')
        self.df[self.y_col] = np.asarray(xylocs[:, 1]).astype('float64')

        # compute other horizontal  movement variables
        self.printit('Computing derived movement variables..')
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

        # vertical movements
        if zlocs is not None:
            self.df[self.z_col] = np.asarray(zlocs).astype('float32')
            zvel = self.df[self.z_col].diff().bfill().divide(tdiff)
            self.df[self.zvel_col] = zvel.astype('float32')

        # local time
        if times_local is not None:
            self.df[self.tlocal_col] = pd.to_datetime(times_local, unit='ms')

        # add other relevant columns'
        self.printit('Cleaning up..')
        self.df[self.region_col] = 'USA' if regions is None else regions
        self.df[self.region_col] = self.df[self.region_col].astype('category')
        self.df[self.region_col] = self.df[self.region_col].cat.remove_unused_categories()
        self.df[self.id_col] = 0 if animalids is None else animalids
        self.df[self.id_col] = self.df[self.id_col].astype('category')
        self.df[self.id_col] = self.df[self.id_col].cat.remove_unused_categories()
        self.df[self.falsefix_col] = False

        # warn about memory usage
        total_mem = np.around(self.df.memory_usage().sum() / 1024 / 1024, 2)
        if total_mem > 1024:
            self.log.warning(f'Memory usage of {total_mem} MB')

    def get_random_track_segment(
        self,
        time_len: float,
        out_dir: str,
        rnd_seed: int = None,
        gf_func=None,
        max_agl: float = 100000,
        **kwargs
    ):
        np.random.seed(rnd_seed)
        track_not_found = True
        while track_not_found:
            itrack = np.random.choice(self.df.TrackID.unique())
            dftrack = self.df[self.df[self.trackid_col] == itrack]
            track_length = dftrack[self.tracktime_col].iloc[-1] - \
                dftrack[self.tracktime_col].iloc[0]
            length = min(time_len, track_length)
            end_row = dftrack.loc[dftrack[self.tracktime_col] >
                                  length, self.tracktime_col].sample(1)
            start_time = end_row.item() - length
            start_index = (dftrack[self.tracktime_col] -
                           start_time).abs().idxmin()
            dfs = dftrack.loc[start_index:end_row.index[0]].copy()
            if dfs['Agl'].quantile(0.75) < max_agl:
                track_not_found = False

        # get 3dep data
        proj_bound = get_bound_from_positions(
            dfs['PositionX'],
            dfs['PositionY'],
            **kwargs
        )
        lonlat_bound = transform_bounds(
            proj_bound, self.proj_crs, self.geo_crs)
        terrain = Data3DEP(
            lonlat_bounds=lonlat_bound,
            resolution=10.,
            out_dir=out_dir
        )
        terrain.download()
        dst = terrain.get_ds(filter_func=gf_func)

        # get hrrr data
        hrrr = DataHRRR(
            time_utc=get_unique_times(dfs['TimeUTC'], scale='h')[0],
            out_dir=self.hrrr_dir
        )
        hrrr.download()
        dst = dst.rio.write_crs(self.proj_crs).rio.reproject(
            ccrs.CRS(hrrr.hrrr_crs),
            resolution=10.,
            nodata=np.nan,
            Resampling=rasterio.enums.Resampling.bilinear
        )
        # ds = dst
        ds = hrrr.ds.isel(time=0).interp(
            x=dst.x,
            y=dst.y,
            method='linear'
        ).merge(dst)
        proj_bound_hrrr = transform_bounds(
            proj_bound,
            self.proj_crs,
            ccrs.CRS(hrrr.hrrr_crs)
        )
        # print(proj_bound_hrrr)
        ds = ds.where(
            (ds.x >= proj_bound_hrrr[0]) & (ds.y >= proj_bound_hrrr[1]) &
            (ds.x <= proj_bound_hrrr[2]) & (ds.y <= proj_bound_hrrr[3]),
            drop=True
        )
        ds['WindSpeed10m'] = np.sqrt(
            ds['WindSpeedU10m']**2 +
            ds['WindSpeedV10m']**2
        )
        ds['WindDirection10m'] = get_wind_direction(
            ds['WindSpeedU10m'],
            ds['WindSpeedV10m']
        )
        ds['WindSpeed80m'] = np.sqrt(
            ds['WindSpeedU80m']**2 +
            ds['WindSpeedV80m']**2
        )
        ds['WindDirection80m'] = get_wind_direction(
            ds['WindSpeedU80m'],
            ds['WindSpeedV80m']
        )
        if gf_func is not None:
            ds['OroSmooth'] = np.cos(np.radians(
                ds['WindDirection80m'])) * ds['OroTerm1']
            ds['OroSmooth'] += np.sin(np.radians(ds['WindDirection80m'])
                                      ) * ds['OroTerm2']
            ds['OroSmooth'] *= ds['WindSpeed80m']
            ds['OroSmooth'] = ds['OroSmooth'].clip(min=0.)
        else:
            oro_updraft = calcOrographicUpdraft_original(
                wspeed=ds['WindSpeed80m'].values,
                wdirn=ds['WindDirection80m'].values,
                slope=ds['GroundSlope'].values,
                aspect=ds['GroundAspect'].values,
                res_terrain=10.,
                res=10.,
                min_updraft_val=1e-5
            )
            ds['Oro'] = (('y', 'x'), oro_updraft)
        ds = ds.drop_vars(['WindSpeedU10m', 'WindSpeedV10m'])
        ds = ds.drop_vars(['WindSpeedU80m', 'WindSpeedV80m'])
        ds = ds.drop_vars(['OroTerm1', 'OroTerm2'])
        southwest = (ds.x.min().item(), ds.y.min().item())
        ds['x'] = ds.x - southwest[0]
        ds['y'] = ds.y - southwest[1]
        dfs['TrackTimeElapsed'] -= dfs['TrackTimeElapsed'].iloc[0]
        xylocs = ccrs.CRS(self.proj_crs).transform_points(
            x=np.asarray(southwest[0]),
            y=np.asarray(southwest[1]),
            src_crs=ccrs.CRS(hrrr.hrrr_crs)
        )
        dfs['PositionX'] -= xylocs[0][0]
        dfs['PositionY'] -= xylocs[0][1]
        return dfs, ds

    def download_3dep_data(
        self,
        resolution: float = 10.,
        ncores: int = mp.cpu_count(),
        verbose: bool = False
    ) -> None:
        """Function for downloading the 3dep data"""
        self.printit('Downloading 3DEP data..[takes longer the first time]')
        list_of_inputs = []
        if self.df_subdomains.empty:
            self.partition_into_subdomains()
        for idx, irow in self.df_subdomains.iterrows():
            lonlat_bnd = irow[['lonmin', 'latmin', 'lonmax', 'latmax']].values
            domain_dir = Path(self.threedep_dir) / idx
            list_of_inputs.append(
                (lonlat_bnd, domain_dir,
                 self.proj_crs.proj4_init, resolution, verbose)
            )
        results = run_loop(
            func=Data3DEP.download_function,
            input_list=list_of_inputs,
            ncores=ncores,
            desc=self.__class__.__name__
        )
        self.df_subdomains['threedep'] = results
        self.printit(f'3DEP data saved in {self.threedep_dir}')

    def download_hrrr_data(
        self,
        ncores: int = mp.cpu_count(),
        tracks_only: bool = True
    ) -> None:
        """Function for downloading the HRRR data"""
        self.printit('Downloading HRRR data..[takes longer the first time]')
        tlist = self.df[self.t_col]
        if tracks_only:
            tlist = self.df.loc[self.df[self.trackid_col] > 0, self.t_col]
        unique_times = get_unique_times(tlist)
        list_of_inputs = []
        for itime in unique_times:
            list_of_inputs.append((itime, self.hrrr_dir))
        _ = run_loop(
            func=DataHRRR.download_function,
            input_list=list_of_inputs,
            ncores=ncores,
            desc=self.__class__.__name__
        )
        self.printit(f'HRRR data saved in {self.hrrr_dir}')

    def annotate_hrrr_data(
        self,
        ncores: int = mp.cpu_count(),
        tracks_only: bool = True
    ):
        """Returns HRRR data for this day and locations"""
        self.download_hrrr_data()
        self.printit(f'Annotating HRRR data using {ncores} cores..')
        unique_days = get_unique_times(self.df[self.t_col], scale='D')
        if tracks_only:
            unique_days = get_unique_times(
                self.df.loc[self.df[self.trackid_col] > 0, self.t_col],
                scale='D'
            )
        locs_hrrr = ccrs.CRS(DataHRRR.hrrr_crs).transform_points(
            x=self.df[self.lon_col].values,
            y=self.df[self.lat_col].values,
            src_crs=ccrs.CRS(self.geo_crs)
        )
        list_of_inputs = []
        for iday in unique_days:
            day_string = np.datetime_as_string(iday, unit='D', timezone='UTC')
            day_string = str(day_string).replace('-', '')
            final_bool = self.df[self.t_col].astype('datetime64[D]') == iday
            if tracks_only:
                final_bool = (final_bool) & (self.df[self.trackid_col] > 0)
            list_of_inputs.append((
                locs_hrrr[final_bool, 0],
                locs_hrrr[final_bool, 1],
                self.df.loc[final_bool, self.t_col].values,
                np.where(final_bool)[0],
                self.hrrr_dir / f'{day_string}*.nc'
            ))
        results = run_loop(
            func=DataHRRR.annotate_function,
            input_list=list_of_inputs,
            ncores=ncores,
            desc=self.__class__.__name__
        )
       # results = self._run(DataHRRR.annotate_function, list_of_inputs, ncores)
        for _, iname in DataHRRR.variables.items():
            self._add_new_column(iname)
        for iresult, ituple in zip(results, list_of_inputs):
            for k, v in iresult.items():
                self.df.loc[ituple[3], k] = np.array(v)
        self.df.sort_index(axis=1, inplace=True)

    def annotate_3dep_data(
        self,
        ncores: int = mp.cpu_count(),
        flag: str = '',
        dist_away: float = 0.,
        angle_away: float = 0.,
        heading_col: str | None = None,
        tracks_only: bool = False,
        filter_func: Callable | None = None
    ) -> None:
        """Annotate 3dep data"""
        self.printit(f'Annotating 3DEP data using {ncores} cores..')
        self.download_3dep_data(verbose=False)
        h_col = self.heading_col if heading_col is None else heading_col
        self.check_validity_of_columns(h_col)
        if tracks_only:
            track_bool = self.df[self.trackid_col] > 0
        list_of_inputs = []
        for _, irow in self.df_subdomains.iterrows():
            ibnd = irow[['lonmin', 'latmin', 'lonmax', 'latmax']].values
            lon_bool = self.df[self.lon_col].between(ibnd[0], ibnd[2])
            lat_bool = self.df[self.lat_col].between(ibnd[1], ibnd[3])
            final_bool = (lon_bool) & (lat_bool)
            if tracks_only:
                final_bool = (final_bool) & (track_bool)
            xlocs = self.df.loc[final_bool, self.x_col].values
            ylocs = self.df.loc[final_bool, self.y_col].values
            hlocs = self.df.loc[final_bool, h_col].values
            xlocs += dist_away * np.sin(np.radians(angle_away + hlocs))
            ylocs += dist_away * np.cos(np.radians(angle_away + hlocs))
            inds = np.where(final_bool)[0]
            if len(inds) > 0:
                list_of_inputs.append((xlocs, ylocs, inds, irow['threedep']))
        fn = partial(Data3DEP.annotate_function, filter_func=filter_func)
        results = run_loop(
            func=fn,
            input_list=list_of_inputs,
            ncores=ncores,
            desc=self.__class__.__name__
        )
        # results = self._run(fn, list_of_inputs, ncores)
        for iname in list(results[0].keys()):
            jname = f'{iname}{flag}'
            # self.printit(f'Annotating {jname} to the dataframe..')
            self._add_new_column(jname)
            for ids, ituple in zip(results, list_of_inputs):
                _, _, inds, _ = ituple
                self.df.loc[inds, jname] = ids[iname].values
        self.df.sort_index(axis=1, inplace=True)

    def _add_new_column(self, iname):
        """Add new column to the dataframe and fill it with nan"""
        if iname not in list(self.df.columns):
            self.df[iname] = np.nan
            self.df[iname] = self.df[iname].astype('float32')

    def annotate_wind_conditions(self, list_of_heights: list[float]):
        """Annotate with wind conditions"""
        for ihgt in list_of_heights:
            istr = f'{str(int(ihgt))}m'
            speed_col = f'WindSpeed{istr}'
            dirn_col = f'WindDirection{istr}'
            angle_col = f'WindRelativeAngle{istr}'
            self.df[speed_col] = np.sqrt(
                self.df[f'WindSpeedU{istr}']**2 +
                self.df[f'WindSpeedV{istr}']**2
            )
            self.df[dirn_col] = get_wind_direction(
                self.df[f'WindSpeedU{istr}'],
                self.df[f'WindSpeedV{istr}']
            )
            self.df[angle_col] = self.df[self.heading_col].subtract(
                180 + self.df[dirn_col])
            self.df.loc[self.df[angle_col] > 180, angle_col] -= 360
            self.df.loc[self.df[angle_col] < -180, angle_col] += 360
            self.df[f'WindSupport{istr}'] = np.cos(np.radians(
                self.df[angle_col])) * self.df[speed_col]
            self.df[f'WindLateral{istr}'] = np.sin(np.radians(
                self.df[angle_col])) * self.df[speed_col]

    def annotate_orographic_updraft(self, flag):
        """annotate orographic updrafts"""
        jname = f'OroSmooth{flag}'
        wdirn = 'WindDirection80m'
        wspeed = 'WindSpeed80m'
        term1 = f'OroTerm1{flag}'
        term2 = f'OroTerm2{flag}'
        self.df[jname] = np.cos(np.radians(self.df[wdirn])) * self.df[term1]
        self.df[jname] += np.sin(np.radians(self.df[wdirn])) * self.df[term2]
        self.df[jname] *= self.df[wspeed]
        # csg.df[f'{jname}_mod'] = expit(2 * (csg.df[jname] - 0.75))
        self.df[jname].clip(lower=0., inplace=True)
        self.df[f'Oro{flag}'] = calcOrographicUpdraft_original(
            wspeed=self.df[wspeed].values,
            wdirn=self.df[wdirn].values,
            slope=self.df[f'GroundSlope{flag}'].values,
            aspect=self.df[f'GroundAspect{flag}'].values,
            res_terrain=10.,
            res=10.,
            min_updraft_val=1e-5
        )

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
        self.df.sort_index(axis=1, inplace=True)
        self.df.reset_index(inplace=True, drop=True)

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
        self.df.loc[track_bool, self.trackid_col] = -1
        track_cat = self.df[self.trackid_col].astype('category')
        self.df[self.trackid_col] = track_cat.cat.codes
        grouped = self.df.groupby(self.trackid_col)[self.tdiff_col]
        self.df[self.tracktime_col] = grouped.transform('cumsum')
        self.df.loc[self.df[self.trackid_col]
                    == 0, self.tracktime_col] = np.nan
        ntracks = self.df.groupby(self.region_col)[self.trackid_col].nunique()
        istr = ', '.join(f'{k}={v}' for k, v in ntracks.to_dict().items())
        self.printit(f'Number of tracks: {istr}')

    def partition_into_subdomains(
        self,
        x_width: float = 20 * 1000,
        y_width: float = 20 * 1000,
        pad: float = 1000.,
        save_domain: bool = False
    ) -> None:
        """Parition spatial data into rectangular chunks"""
        self.check_validity_of_columns([self.x_col, self.y_col])
        self.df[self.domain_col] = -1
        self.df_subdomains = {}
        for iregion in self.df[self.region_col].unique():
            rbool = self.df[self.region_col] == iregion
            x_edges = get_bin_edges(
                self.df.loc[rbool, self.x_col], x_width, pad)
            y_edges = get_bin_edges(
                self.df.loc[rbool, self.y_col], y_width, pad)
            x_bins = np.digitize(self.df.loc[rbool, self.x_col], x_edges)
            y_bins = np.digitize(self.df.loc[rbool, self.y_col], y_edges)
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
            istr = f'{iregion} = {uniq.shape[1]}/{num_domains}'
            self.printit(f'Region {istr} domains contain data')
        self.df_subdomains = pd.DataFrame.from_dict(
            self.df_subdomains,
            orient='index',
            columns=['xmin', 'xmax', 'ymin', 'ymax']
        )
        geo_corners = ccrs.CRS(self.geo_crs).transform_points(
            x=self.df_subdomains[['xmin', 'xmin', 'xmax', 'xmax']].values,
            y=self.df_subdomains[['ymin', 'ymax', 'ymin', 'ymax']].values,
            src_crs=ccrs.CRS(self.proj_crs)
        )
        self.df_subdomains['lonmin'] = np.amin(geo_corners, axis=1)[:, 0]
        self.df_subdomains['lonmax'] = np.amax(geo_corners, axis=1)[:, 0]
        self.df_subdomains['latmin'] = np.amin(geo_corners, axis=1)[:, 1]
        self.df_subdomains['latmax'] = np.amax(geo_corners, axis=1)[:, 1]
        self.df_subdomains.index.name = self.domain_col
        self.df[self.domain_col] = self.df[self.domain_col].astype('category')
        if save_domain:
            self.df_subdomains.to_pickle(self.domain_fpath)

    def ignore_data_based_on_vertical_speed(
        self,
        max_change: float,
        max_abs_val: float = None
    ) -> None:
        """False fix points based on vertical speed"""
        assert self.zvel_col in self.df.columns, f'Vertical data not loaded!'
        vspeed = self.df[self.zvel_col]
        vchange = vspeed - np.maximum(vspeed.shift(1), vspeed.shift(-1))
        cond1 = (vspeed >= 0.) & (vchange >= max_change)
        vchange = np.minimum(vspeed.shift(1), vspeed.shift(-1)) - vspeed
        cond2 = (vspeed <= 0.) & (vchange >= max_change)
        cond_both = (cond1) | (cond2)
        self.df[self.falsefix_col] = self.df[self.falsefix_col] | cond_both
        if max_abs_val is not None:
            condn = (self.df[self.zvel_col].abs() > max_abs_val)
            self.df[self.falsefix_col] = self.df[self.falsefix_col] | condn
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

    def df_track(self, track_id: int):
        """Returns track df"""
        return self.df[self.df[self.trackid_col] == track_id]

    def info(self):
        """Returns the compresses info of the dataframe containing the data"""
        self.df.info(verbose=True, memory_usage=True, show_counts=True)

    @ property
    def track_ids(self):
        """return a list of tracks"""
        out_list = []
        if self.trackid_col in self.df.columns:
            out_list = list(self.df[self.trackid_col].unique())
        return out_list
