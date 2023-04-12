
""" Utility functions """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
from typing import Callable, Sequence
from functools import partial
from dataclasses import dataclass, field
import time
import numpy as np
import tqdm
import pandas as pd
# import pathos.multiprocessing as mp
from numpy import ndarray
#from .telemetry_data import TelemetryData
# from .kalman_resampler import KalmanTrackResampler


def get_bin_edges(locs, width, pad):
    """returns edges from bins"""
    ds = pd.Series(locs)
    bnd = [ds.min() - pad, ds.max() + pad]
    count = int(np.ceil((bnd[1] - bnd[0]) / width))
    return np.linspace(bnd[0], bnd[0] + width * count, count + 1)


def get_unique_times(
    list_of_times: list[np.datetime64],
    list_of_lags: list[int] | None = None,
    time_scale: str = 'h'
) -> list[np.datetime64]:
    """
    Function to get unique time instances contained in the telemetry
    data available in a pandas datarfame

    Parameters
    ----------
    idf: pd.DataFrame
        Pandas dataframe containing the telemetry data
    hourly_lags: List[int]
        List of time lags to consider when figuring out unique times
        for instance, for hourly time scale, for 3:23 PM and hourly_lag of 0
        means 3 PM and hourly lag of 1 means 4 PM
    time_colname: str
        Name of the column containing time stamps, UTC or local time,
        should be in numpy datetime64 format,
        or in format that pd.to_datetime likes
    time_scale: str
        Could be hourly 'h', daily 'D', monthly 'M' or others
        see https://numpy.org/doc/stable/reference/arrays.datetime.html


    Returns
    -------
    List[np.datetime64]
        timestamps at hourly reso
    """

    # get all unique times at the lower bound of each time
    # unique_times = np.array([np.datetime64(itime) for itime in list_of_times])
    unique_times = list_of_times.astype(f'datetime64[{time_scale}]')
    unique_times = np.unique(unique_times)
    # include instances before and after
    list_of_lags = [0] if list_of_lags is None else list_of_lags
    list_of_lags = [list_of_lags] if not isinstance(
        list_of_lags, Sequence) else list_of_lags
    for ilag in list_of_lags:
        time_lagged = unique_times + np.timedelta64(int(ilag), time_scale)
        unique_times = np.hstack([unique_times, time_lagged])
    unique_times = np.sort(np.unique(unique_times))
    unique_times = unique_times.astype(f'datetime64[{time_scale}]')
    return unique_times


def get_wind_direction(uspeed, vspeed):
    """return wind speed and direction"""
    dirn = np.degrees(np.arctan2(uspeed, vspeed))
    return dirn - np.sign(dirn) * 180.

# def plot_usa_map(ax, proj_crs):
#     """Plots usa map in a given projection system"""
#     usa = gpd.read_file(os.path.join(
#         data_dir, 'maps', 'usa_states', 'cb_2018_us_state_20m.shp'))
#     usa_crs = usa.to_crs(crs=csg.geo_crs)
#     fig, ax = plt.subplots(figsize=(8, 8))
#     usa_crs.boundary.plot(ax=ax, linewidth=0.3)

# def resample_all_tracks(
#     tdata: TelemetryData,
#     sampler: KalmanTrackResampler,
#     num_cores: int = 8
# ):
#     list_of_track_dfs = [tdata.df_track(ix) for ix in tdata.track_ids]
#     with mp.ProcessPool(num_cores) as pool:
#         out_list = tqdm.tqdm(pool.imap(
#             kf_ca1d, list_of_track_dfs),
#             total=len(list_of_track_dfs)
#         )
#     rdf = pd.concat(out_list, axis=0)

#     # Get lat lon and save the resampled data
#     print('\nSaving the data..', flush=True)
#     rdf.rename(columns={
#         'ObjectId': 'TrackID',
#         'TimeElapsed': 'TrackTimeElapsed'
#     }, inplace=True)
#     xlocs, ylocs = transform_coordinates(TELEMETRY_CRS, GEO_CRS,
#                                          rdf['PositionX'].values,
#                                          rdf['PositionY'].values)
#     rdf['Longitude'] = np.array(xlocs, dtype='float32')
#     rdf['Latitude'] = np.array(ylocs, dtype='float32')

#     # other derived variables
#     rdf['VelocityHor'] = np.sqrt(rdf['VelocityX']**2 + rdf['VelocityY']**2)
#     rdf['HeadingHor'] = np.arctan2(rdf['VelocityX'], rdf['VelocityY'])
#     rdf['HeadingHor'] = (np.degrees(rdf['HeadingHor'])) % 360
#     rdf['AccnHorTangential'] = (rdf['AccelerationX'] * rdf['VelocityX'] +
#                                 rdf['AccelerationY'] * rdf['VelocityY'])
#     rdf['AccnHorTangential'] = rdf['AccnHorTangential'] / rdf['VelocityHor']
#     rdf['AccnHorRadial'] = (rdf['AccelerationY'] * rdf['VelocityX'] -
#                             rdf['AccelerationX'] * rdf['VelocityY'])
#     rdf['AccnHorRadial'] = rdf['AccnHorRadial'] / rdf['VelocityHor']
#     rdf['RadiusOfCurvature'] = rdf['VelocityHor']**2 / \
#         rdf['AccnHorRadial'].abs()
#     rdf['HeadingRateHor'] = np.degrees(
#         rdf['AccnHorRadial'] / rdf['VelocityHor'])
#     for icol in ['Age', 'Sex', 'Group']:
#         rdf[icol] = rdf[icol].astype("category")
#     rdf.to_pickle(CRATE_FILEPATH)
