
""" Utility functions """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
import sys
import time
from typing import Sequence
import numpy as np
import xarray as xr
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import pathos.multiprocessing as mp
import matplotlib.pyplot as plt
# import rasterio


def add_angles(angle1, angle2):
    """add two arctan2 returned angles"""
    diff = angle1 + angle2
    diff = diff - 360 if diff > 180 else diff
    diff = diff + 360 if diff < -180 else diff
    return diff


def run_loop(func, input_list, ncores=mp.cpu_count(), **kwargs):
    """Run parallel simulation"""
    pbar = tqdm(
        iterable=input_list,
        total=len(input_list),
        position=0,
        leave=True,
        file=sys.stdout,
        **kwargs
    )
    if ncores <= 1:
        results = []
        for ix in pbar:
            results.append(func(ix))
    else:
        with mp.Pool(ncores) as pool:
            results = list(pool.imap(func, pbar))
    return results


def get_bin_edges(locs, width, pad):
    """returns edges from bins"""
    ds = pd.Series(locs)
    bnd = [ds.min() - pad, ds.max() + pad]
    count = int(np.ceil((bnd[1] - bnd[0]) / width))
    return np.linspace(bnd[0], bnd[0] + width * count, count + 1)


def get_unique_times(
    times: list[np.datetime64],
    lags: list[int] | None = None,
    scale: str = 'h'
) -> list[np.datetime64]:
    """
    Function to get unique time instances contained in the telemetry
    data available in a pandas datarfame

    Parameters
    ----------
    idf: pd.DataFrame
        Pandas dataframe containing the telemetry data
    list_of_lags: List[int]
        List of time lags to consider when figuring out unique times
        for instance, for hourly time scale, for 3:23 PM and hourly_lag of 0
        means 3 PM and hourly lag of 1 means 4 PM
    time_scale: str
        Could be hourly 'h', daily 'D', monthly 'M' or others
        see https://numpy.org/doc/stable/reference/arrays.datetime.html
    Returns
    -------
    List[np.datetime64]
        timestamps at hourly reso
    """

    # get all unique times at the lower bound of each time
    # utimes = np.array([np.datetime64(itime) for itime in list_of_times])
    utimes = times.values if isinstance(times, pd.Series) else times
    utimes = utimes.astype(f'datetime64[{scale}]')
    utimes = np.unique(utimes)
    # include instances before and after
    list_of_lags = [0] if lags is None else lags
    list_of_lags = [list_of_lags] if not isinstance(
        list_of_lags, Sequence) else list_of_lags
    for ilag in list_of_lags:
        time_lagged = utimes + np.timedelta64(int(ilag), scale)
        utimes = np.hstack([utimes, time_lagged])
    utimes = np.sort(np.unique(utimes))
    utimes = utimes.astype(f'datetime64[{scale}]')
    # utimes = utimes.item() if utimes.size == 1 else utimes
    return utimes


def get_wind_direction(uspeed, vspeed):
    """return wind speed and direction"""
    dirn = np.degrees(np.arctan2(uspeed, vspeed))
    return dirn - np.sign(dirn) * 180.


def plot_relation(
    idf,
    yname,
    xnames,
    quant=0.99,
    fig=None,
    ax=None,
    clr='b',
    x_std: bool = True,
    y_std: bool = True,
    lags: bool = False,
    ylim=None
):
    lbls = {
        'Agl': 'Altitude AGL F0 [m]',
        'AglCubic': 'Altitude AGL cubic [m]',
        'AglMod': 'Altitude AGL F0 Mod [m]',
        'AglLog': 'Altitude AGL F0 Log [m]',
        'AglNear': 'Altitude AGL F50 [m]',
        'AglClose': 'Altitude AGL F250 [m]',
        'AglMid': 'Altitude AGL F500 [m]',
        'AglFar': 'Altitude AGL F1000 [m]',
        'AglFarther': 'Altitude AGL F2000 [m]',
        'ElevNearDiff': 'Elev F50-F0 [m]',
        'ElevCloseDiff': 'Elev F250-F0 [m]',
        'ElevMidDiff': 'Elev F500-F0 [m]',
        'ElevFarDiff': 'Elev F1000-F0 [m]',
        'ElevFartherDiff': 'Elev F2000-F0 [m]',

        'OroSmooth': 'Orographic Updraft PF [m/s]',
        'OroSmoothNearDiff': 'Orographic Updraft F50-F0 [m/s]',
        'OroSmoothCloseDiff': 'Orographic Updraft F250-F0 [m/s]',
        'OroSmoothMidDiff': 'Orographic Updraft F500-F0 [m/s]',
        'OroSmoothFarDiff': 'Orographic Updraft F1000-F0 [m/s]',
        'OroSmoothFartherDiff': 'Orographic Updraft F2000-F0 [m/s]',

        'OroSmoothNearL30Diff': 'Orographic Updraft F50L30-F50 [m/s]',
        'OroSmoothNearR30Diff': 'Orographic Updraft F50R30-F50 [m/s]',
        'OroSmoothNearL60Diff': 'Orographic Updraft F50L60-F50 [m/s]',
        'OroSmoothNearR60Diff': 'Orographic Updraft F50R60-F50 [m/s]',

        'OroSmoothCloseL30Diff': 'Orographic Updraft F250L30-F250 [m/s]',
        'OroSmoothCloseR30Diff': 'Orographic Updraft F250R30-F250 [m/s]',
        'OroSmoothCloseL60Diff': 'Orographic Updraft F250L60-F250 [m/s]',
        'OroSmoothCLoseR60Diff': 'Orographic Updraft F250R60-F250 [m/s]',

        'HeadingRateAbs': 'Heading Rate abs [deg/s]',
        'VelocityHor': 'Horizontal Speed [m/s]',
        'VelocityVer': 'Vertical Speed [m/s]',
        'WindSpeed80m': 'Wind Speed [m/s]',
        'WindSupport80m': 'Wind Support [m/s]',
        'WindLateral80m': 'Wind Lateral [m/s]',
        'WindLateral80mAbs': 'Wind Lateral, Abs [m/s]',
        'VelocityHorNext': 'Horizontal Speed [m/s]',
        'VelocityVerNext': 'Vertical Speed [m/s]',
        'HeadingRateNext': 'Heading Rate [deg/s]',
        'VelocityHorLag5': 'Horizontal Speed 5s ago [m/s]',
        'VelocityVerLag5': 'Vertical Speed 5s ago [m/s]',
        'HeadingRateLag5': 'Heading Rate 5s ago [deg/s]',
        'VelocityHorLag10': 'Horizontal Speed 10s ago [m/s]',
        'VelocityVerLag10': 'Vertical Speed 10s ago [m/s]',
        'HeadingRateLag10': 'Heading Rate 10s ago [deg/s]',
        'VelocityHorLag20': 'Horizontal Speed 20s ago [m/s]',
        'VelocityVerLag20': 'Vertical Speed 20s ago [m/s]',
        'HeadingRateLag20': 'Heading Rate 20s ago [deg/s]',
        'VelocityHorLag30': 'Horizontal Speed 30s ago [m/s]',
        'VelocityVerLag30': 'Vertical Speed 30s ago [m/s]',
        'HeadingRateLag30': 'Heading Rate 30s ago [deg/s]',
        'VelocityHorLag60': 'Horizontal Speed 60s ago [m/s]',
        'VelocityVerLag60': 'Vertical Speed 60s ago [m/s]',
        'HeadingRateLag60': 'Heading Rate 60s ago [deg/s]',
        'VelocityHorLag120': 'Horizontal Speed 120s ago [m/s]',
        'VelocityVerLag120': 'Vertical Speed 120s ago [m/s]',
        'HeadingRateLag120': 'Heading Rate 120s ago [deg/s]',

    }
    if not lags:
        xnames = [ix for ix in xnames if 'Lag' not in ix]
    if fig is None:
        nrows = int(len(xnames) // 3)
        nrows += 1 if len(xnames) % 3 > 0 else 0
        fig, ax = plt.subplots(nrows, 3, figsize=(
            12, 2.25 * nrows))
        ax = ax.flatten()
    ynameb = yname.split('Std')[0] if not y_std else yname
    for i, (iax, xname) in enumerate(zip(ax, xnames)):
        xnameb = xname.split('Std')[0] if not x_std else xname
        xgrid = np.linspace(*np.quantile(idf[xnameb], [1 - quant, quant]), 50)
        rdfshortb = pd.DataFrame({'x': idf[xnameb], 'y': idf[ynameb]})
        rdfshortb['bin_col'] = np.searchsorted(xgrid, idf[xnameb])
        bin_groups = rdfshortb.groupby(by='bin_col')
        iax.plot(xgrid, bin_groups['y'].mean()[:-1],
                 linestyle='-', color=clr, linewidth=2., alpha=0.75)
        if xname.split('Std')[0] in lbls.keys():
            iax.set_xlabel(lbls[xname.split('Std')[0]])
        if i % 3 == 0:
            if yname.split('Std')[0] in lbls.keys():
                iax.set_ylabel(lbls[yname.split('Std')[0]])
            if yname.split('Next')[0] in lbls.keys():
                iax.set_ylabel(lbls[yname.split('Next')[0]])
        iax.set_xlim(np.quantile(idf[xname], [1 - quant, quant]))
        if ylim is not None:
            iax.set_ylim(ylim)
        iax.grid(True)
    for i in range(len(xnames), len(ax)):
        ax[i].axis('off')
    fig.tight_layout()
    return fig, ax


def plot_2drelation(
    idf,
    xname,
    yname,
    quant=0.99,
    nsamples=2000,
    fig=None,
    ax=None,
    scatter=False,
    clr='b'
):
    if fig is None:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3))
        ax = ax.flatten()
    idf[xname].hist(ax=ax[0], bins=100, histtype='step',
                    label=xname, density=True)
    idf[yname].hist(ax=ax[0], bins=100, histtype='step',
                    label=yname, density=True)
    ax[0].legend()
    ax[0].set_yticks([])
    # ax[0].plot(idf[xname], idf[yname], '.b', markersize=0.05, alpha=0.25)
    xgrid = np.linspace(*np.quantile(idf[xname], [1 - quant, quant]), 50)
    rdfshortb = pd.DataFrame({'x': idf[xname], 'y': idf[yname]})
    rdfshortb['bin_col'] = np.searchsorted(xgrid, idf[xname])
    bin_groups = rdfshortb.groupby(by='bin_col')
    if scatter:
        for _, idata in bin_groups:
            slice_data = idata.sample(nsamples, replace=True)
            ax[1].plot(slice_data['x'], slice_data['y'],
                       f'.{clr}', markersize=0.2, alpha=0.25)
    ax[1].plot(xgrid, bin_groups['y'].mean()[:-1],
               linestyle='-', color=clr, linewidth=2., alpha=0.75)
    # ax[1].plot(xgrid, bin_groups['y'].median()[:-1],
    #            linestyle='-.', color=clr, linewidth=1., alpha=0.75)
    ax[1].set_xlabel(xname)
    ax[1].set_ylabel(yname)
    # ax[1].set_xlim(np.quantile(idf[xname], [1 - quant, quant]))
    # ax[1].set_ylim(np.quantile(idf[yname], [1 - quant, quant]))
    ax[1].grid(True)

    ax[2].plot(xgrid, bin_groups['y'].std()[:-1],
               linestyle='-', color=clr, linewidth=2., alpha=0.75)
    ax[2].set_ylabel(f'var({yname})')
    ax[2].set_xlabel(xname)
    ax[2].grid(True)
    ax[2].set_xlim(np.quantile(idf[xname], [1 - quant, quant]))
    ax[2].set_ylim([0, 2])
    fig.tight_layout()
    return fig, ax


# def get_fitted_model(, covariates, predictable, poly_func, regressors):
#     """return fitted model"""

#     start_time = time.time()
#     fitted_model = regressor.fit(Xmat, Yvec)
#      models[reg_name] = deepcopy(fitted_model)
#       coeff_val = np.around(fitted_model.coef_[:], 5)
#        # model_score = np.around(fitted_model.score(Xmat, Yvec), 3)
#        for i, val in enumerate(coeff_val):
#             coeff_val[i] = val if abs(val) > 1e-10 else np.nan
#         result_df[f'{reg_name}_{predictable}'] = list(
#             coeff_val) + [fitted_model.intercept_, fitted_model.score(Xmat, Yvec)]
#         run_time = np.around(((time.time() - start_time)) / 60., 2)
#         print(f'{reg_name}-{predictable}-took {run_time} mins', flush=True)
#     models['df'] = pd.DataFrame(result_df)
#     return models


# def get_fitted_model(xdf, covariates, predictable, poly_func, regressors):
#     """return fitted model"""
#     models = {}
#     models['covariates'] = covariates
#     Xmat = poly_func.fit_transform(xdf.loc[:, covariates].values)
#     poly_func.feature_names_in_ = covariates
#     models['poly_func'] = poly_func
#     result_df = pd.DataFrame(index=list(
#         poly_func.get_feature_names_out()) + ['Bias', 'Score'])
#     for reg_name, regressor in regressors.items():
#         start_time = time.time()
#         Yvec = xdf.loc[:, predictable].values
#         fitted_model = regressor.fit(Xmat, Yvec)
#         models[reg_name] = deepcopy(fitted_model)
#         coeff_val = np.around(fitted_model.coef_[:], 5)
#         # model_score = np.around(fitted_model.score(Xmat, Yvec), 3)
#         for i, val in enumerate(coeff_val):
#             coeff_val[i] = val if abs(val) > 1e-10 else np.nan
#         result_df[f'{reg_name}_{predictable}'] = list(
#             coeff_val) + [fitted_model.intercept_, fitted_model.score(Xmat, Yvec)]
#         run_time = np.around(((time.time() - start_time)) / 60., 2)
#         print(f'{reg_name}-{predictable}-took {run_time} mins', flush=True)
#     models['df'] = pd.DataFrame(result_df)
#     return models


def get_bound_from_positions(
    xlocs,
    ylocs,
    xpad_km: float,
    ypad_km: float,
    min_xwidth_km: float,
    min_ywidth_km: float
):
    """get terrain bounds"""
    proj_bounds = [
        np.amin(xlocs) - xpad_km * 1000., np.amin(ylocs) - ypad_km * 1000.,
        np.amax(xlocs) + xpad_km * 1000., np.amax(ylocs) + ypad_km * 1000.
    ]
    xwidth = max(proj_bounds[2] - proj_bounds[0], min_xwidth_km * 1000.)
    ywidth = max(proj_bounds[3] - proj_bounds[1], min_ywidth_km * 1000.)
    width = max(xwidth, ywidth)
    center = [(proj_bounds[2] + proj_bounds[0]) / 2.,
              (proj_bounds[1] + proj_bounds[3]) / 2.]
    proj_bounds = [center[0] - xwidth / 2, center[1] - ywidth / 2,
                   center[0] + xwidth / 2, center[1] + ywidth / 2]
    proj_bounds = [center[0] - width / 2, center[1] - width / 2,
                   center[0] + width / 2, center[1] + width / 2]
    return proj_bounds


# def get_terrain_ds(
#     lonlat_bound,
#     wind_conditions,
#     resolution,
#     gf_sigma,
#     out_dir
# ):
#     TELEMETRY_CRS = 'ESRI:102008'
#     GEO_CRS = 'EPSG:4326'
#     DEM_3DEP_LAYER = 'DEM'
#     region = Terrain(lonlat_bound, out_dir, print_verbose=False)
#     region.download(DEM_3DEP_LAYER)
#     dsg = rio.open_rasterio(region.get_raster_fpath(DEM_3DEP_LAYER))
#     dsg = dsg.to_dataset(name='GroundElevation').squeeze()
#     ds = self.ds_geo.rio.reproject(
#         TELEMETRY_CRS,
#         resolution=resolution,
#         nodata=np.nan,
#         Resampling=rasterio.enums.Resampling.bilinear
#     )
#     ds['WindSpeed'] = xr.DataArray(
#         data=wind_conditions[0],
#         dims=('y', 'x'),
#         coords=dict(y=ds.y.values, x=ds.x.values)
#     )
#     ds['WindDirection'] = xr.DataArray(
#         data=wind_conditions[1],
#         dims=('y', 'x'),
#         coords=dict(y=ds.y.values, x=ds.x.values)
#     )
#     oro_updraft = calcOrographicUpdraft_original(
#         wspeed=ds['WindSpeed'].values,
#         wdirn=ds['WindDirection'].values,
#         slope=ds['GroundSlope'].values,
#         aspect=ds['GroundAspect'].values,
#         res_terrain=10.,
#         res=10.
#     )
#     ds['OroUpdraft'] = xr.DataArray(
#         data=gaussian_filter(oro_updraft, sigma=gf_sigma),
#         dims=('y', 'x'),
#         coords=dict(y=ds.y.values, x=ds.x.values)
#     )
#     ds['OroUpdraftS'] = xr.DataArray(
#         data=get_std_orographic_updraft(ds['OroUpdraft'].values),
#         dims=('y', 'x'),
#         coords=dict(y=ds.y.values, x=ds.x.values)
#     )
#     return ds


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
