"""CSG Data processing script"""

import os
import time
from functools import partial
import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from bayesfilt.telemetry import Telemetry
from bayesfilt.telemetry import Data3DEP, DataHRRR
from bayesfilt.telemetry.utils import get_wind_direction
from ssrs.layers import calcOrographicUpdraft_original
from bayesfilt.telemetry import ConstantVelocityResampler, KalmanResampler
from bayesfilt.telemetry import ConstantAccelerationResampler
from bayesfilt.telemetry import CorrelatedVelocityResampler
from bayesfilt.models import CVM3D_NL_4, LinearObservationModel
from bayesfilt.filters import UnscentedKalmanFilter
from bayesfilt.telemetry import BaseGeoData
from bayesfilt.telemetry.utils import run_loop


def resample_function_cvm(dftrack):
    """resample function"""
    # phi = {
    #     'eta_hor': 3.,
    #     'sigma_log_tau_hor': 0.1,
    #     'sigma_mu_hor': 0.2,
    #     'sigma_omega': 0.01,
    #     'eta_ver': 1.,
    #     'sigma_log_tau_ver': 0.1,
    #     'sigma_mu_ver': 0.04
    # }
    phi = {
        'eta_hor': 1.,
        'sigma_log_tau_hor': 0.1,
        'sigma_mu_hor': 0.2,
        'sigma_omega': 0.01,
        'eta_ver': 1.,
        'sigma_log_tau_ver': 0.1,
        'sigma_mu_ver': 0.04
    }
    cvm = CorrelatedVelocityResampler(phi=phi, dt=1., smoother=True)
    cvm.resample(dftrack)
    try:
        out_df = postprocess_dframe(cvm.kf.dfs, dftrack, cvm.mm.dt)
    except:
        out_df = pd.DataFrame({})
    return out_df


def resample_function_ca(dftrack):
    """resample function"""
    track_id = dftrack['TrackID'].iloc[0]
    sampler = partial(ConstantAccelerationResampler, dt=1., smoother=True)
    rs_x = sampler(error_strength=0.75, flag='X')
    rs_x.resample(
        times=dftrack['TrackTimeElapsed'].values,
        locs=dftrack['PositionX'].values,
        error_std=dftrack['ErrorHDOP'].values * 2.5 / np.sqrt(2),
        start_state_std=[4., 2., 2.],
        object_id=track_id
    )
    rs_y = sampler(error_strength=0.75, flag='Y')
    rs_y.resample(
        times=dftrack['TrackTimeElapsed'].values,
        locs=dftrack['PositionY'].values,
        error_std=dftrack['ErrorHDOP'].values * 2.5 / np.sqrt(2),
        start_state_std=[4., 2., 2.],
        object_id=track_id
    )
    rs_z = sampler(error_strength=0.25, flag='Z')
    rs_z.resample(
        times=dftrack['TrackTimeElapsed'].values,
        locs=dftrack['Altitude'].values,
        error_std=dftrack['ErrorVDOP'].values * 4.0,
        start_state_std=[4., 2., 2.],
        object_id=track_id
    )
    sdf = pd.concat((rs_x.kf.dfs, rs_y.kf.dfs, rs_z.kf.dfs),
                    axis=1, join='outer')
    return postprocess_dframe(sdf, dftrack, rs_x.mm.dt)


def resample_function_cv(dftrack):
    """resample function"""
    track_id = dftrack['TrackID'].iloc[0]
    sampler = partial(ConstantVelocityResampler, dt=1., smoother=True)
    rs_x = sampler(error_strength=1.5, flag='X')
    rs_x.resample(
        times=dftrack['TrackTimeElapsed'].values,
        locs=dftrack['PositionX'].values,
        error_std=dftrack['ErrorHDOP'] * 2.5,
        start_state_std=[5., 2.],
        object_id=track_id
    )
    rs_y = sampler(error_strength=1.5, flag='Y')
    rs_y.resample(
        times=dftrack['TrackTimeElapsed'],
        locs=dftrack['PositionY'],
        error_std=dftrack['ErrorHDOP'] * 2.5,
        start_state_std=[5., 2.],
        object_id=track_id
    )
    rs_z = sampler(error_strength=0.5, flag='Z')
    rs_z.resample(
        times=dftrack['TrackTimeElapsed'],
        locs=dftrack['Altitude'],
        error_std=dftrack['ErrorVDOP'] * 4.5,
        start_state_std=[5., 2.],
        object_id=track_id
    )
    sdf = pd.concat((rs_x.kf.dfs, rs_y.kf.dfs, rs_z.kf.dfs),
                    axis=1, join='outer')
    return postprocess_dframe(sdf, dftrack, rs_x.mm.dt)


def postprocess_dframe(sdf, dftrack, dt):
    """postprocess function"""
    sdf = sdf.loc[:, ~sdf.columns.duplicated()].copy()
    #cols_to_drop = [ix for ix in sdf.columns if '_var' in ix]
    #cols_to_drop = [ix for ix in cols_to_drop if 'Position' not in ix]
    cols_to_drop = [ix for ix in sdf.columns if 'Metric' in ix]
    sdf.drop(columns=cols_to_drop, inplace=True)
    sdf['TimeUTC'] = pd.date_range(
        start=dftrack['TimeUTC'].iloc[0],
        periods=len(sdf),
        freq=str(dt) + "s"
    )
    sdf['TimeLocal'] = pd.date_range(
        start=dftrack['TimeLocal'].iloc[0],
        periods=len(sdf),
        freq=str(dt) + "s"
    )
    sdf['Group'] = [dftrack['Group'].iloc[0]] * len(sdf)
    sdf['AnimalID'] = [dftrack['AnimalID'].iloc[0]] * len(sdf)
    sdf['Age'] = [dftrack['Age'].iloc[0]] * len(sdf)
    sdf['Sex'] = [dftrack['Sex'].iloc[0]] * len(sdf)
    return sdf


def annotate_derived_vars(rdf, tdf):
    """Annotate other derived movement variables"""
    rdf = rdf.rename(columns={
        'ObjectId': 'TrackID',
        'TimeElapsed': 'TrackTimeElapsed',
        'AccelerationZ': 'AccelerationVer',
        'VelocityZ': 'VelocityVer',
        'VelocityZ_var': 'VelocityVer_var',
        'PositionZ': 'Altitude',
        'PositionZ_var': 'Altitude_var'
    })
    rdf['Group'] = rdf['Group'].astype('category')
    rdf['AnimalID'] = rdf['AnimalID'].astype('category')
    rdf['Age'] = rdf['Age'].astype('category')
    rdf['Sex'] = rdf['Sex'].astype('category')
    geo_obj = BaseGeoData(proj_crs='ESRI:102008')
    xylocs = ccrs.CRS(geo_obj.geo_crs).transform_points(
        x=rdf['PositionX'].values,
        y=rdf['PositionY'].values,
        src_crs=geo_obj.proj_crs
    )
    rdf['Longitude'] = np.asarray(xylocs[:, 0]).astype('float32')
    rdf['Latitude'] = np.asarray(xylocs[:, 1]).astype('float32')
    rdf['VelocityHor'] = np.sqrt(rdf['VelocityX']**2 + rdf['VelocityY']**2)
    rdf['HeadingHor'] = np.arctan2(rdf['VelocityX'], rdf['VelocityY'])
    #rdf['HeadingHor'] = (np.degrees(rdf['HeadingHor'])) % 360
    rdf.reset_index(inplace=True, drop=True)
    xbool = rdf['PositionX'].between(
        tdf['PositionX'].min(),
        tdf['PositionX'].max()
    )
    ybool = rdf['PositionY'].between(
        tdf['PositionY'].min(),
        tdf['PositionY'].max()
    )
    rdf = rdf.loc[(xbool) & (ybool), :]
    if 'AccelerationX' in rdf.columns:
        rdf['AccnHorTangential'] = (rdf['AccelerationX'] * rdf['VelocityX'] +
                                    rdf['AccelerationY'] * rdf['VelocityY'])
        rdf['AccnHorTangential'] = rdf['AccnHorTangential'] / rdf['VelocityHor']
        rdf['AccnHorRadial'] = (rdf['AccelerationY'] * rdf['VelocityX'] -
                                rdf['AccelerationX'] * rdf['VelocityY'])
        rdf['AccnHorRadial'] = rdf['AccnHorRadial'] / rdf['VelocityHor']
        rdf['RadiusOfCurvature'] = rdf['VelocityHor']**2 / \
            rdf['AccnHorRadial'].abs()
        rdf['HeadingRateHor'] = np.degrees(
            rdf['AccnHorRadial'] / rdf['VelocityHor'])
    return rdf


if __name__ == "__main__":

    print('\n---Data resampling script', flush=True)
    start_time = time.time()

    output_dir = os.path.join('/home/rsandhu/zazzle/BSSRS/IpcFusion/examples/csg/output')
    FNAME = 'csg_ge_vr.prq_tracks'
    vrate_fpath = os.path.join(output_dir, 'telemetry', FNAME)
    df = pd.read_parquet(vrate_fpath)
    df_bool = df['Group'].isin(['pa', 'wy', 'hr'])
    list_of_tracks = df.loc[df_bool, 'TrackID'].unique()
    #list_of_tracks = csg.df['TrackID'].unique()
    list_of_dftrack = [df[df['TrackID'] == ix]
                       for ix in list_of_tracks if ix != 0]

    # # run cv resampler
    # results = run_loop(
    #     func=resample_function_cv,
    #     input_list=list_of_dftrack,
    #     desc='CVresampler'
    # )
    # #results = run_loop(resample_function_cv, list_of_dftrack)
    # rsdf = pd.concat(results)
    # rsdf = annotate_derived_vars(rsdf, df)
    # rs_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_cv')
    # print(f'Saving to {rs_fpath}', flush=True)
    # rsdf.to_parquet(rs_fpath)

    # run ca resampler
    results = run_loop(
        func=resample_function_ca,
        input_list=list_of_dftrack,
        desc='CAresampler'
    )
    rsdf = pd.concat(results)
    rsdf = annotate_derived_vars(rsdf, df)
    rs_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_ca')
    print(f'Saving to {rs_fpath}', flush=True)
    rsdf.to_parquet(rs_fpath)

    # # run cvm resampler
    # rs_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_ca')
    # rsdf = pd.read_parquet(rs_fpath)
    # list_of_dftrack = [rsdf[rsdf['TrackID'] == ix]
    #                    for ix in list_of_tracks if ix != 0]
    # results = run_loop(
    #     func=resample_function_cvm,
    #     input_list=list_of_dftrack,
    #     desc='CVMresampler'
    # )
    # rsdf = pd.concat(results)
    # rsdf = annotate_derived_vars(rsdf, df)
    # rs_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_cvm')
    # print(f'Saving to {rs_fpath}', flush=True)
    # rsdf.to_parquet(rs_fpath)

    # end
    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)
