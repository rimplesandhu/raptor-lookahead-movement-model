"""CSG Data processing script"""

import os
import time
from functools import partial
import pandas as pd
import numpy as np
import xarray as xr
from scipy.special import expit
from scipy.ndimage import gaussian_filter
from bayesfilt.telemetry import Telemetry
from bayesfilt.telemetry import Data3DEP, DataHRRR
from bayesfilt.telemetry.utils import get_wind_direction
from ssrs.layers import calcOrographicUpdraft_original


if __name__ == "__main__":

    print('\n---Data annotation script', flush=True)
    start_time = time.time()

    output_dir = os.path.join('/home/rsandhu/projects_car/csg_data/output')
    FNAME = 'csg_ge_vr.prq'
    vrate_fpath = os.path.join(output_dir, 'telemetry', FNAME)
    ann_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_tracks')
    vdf = pd.read_parquet(vrate_fpath)

    vdf_bool = vdf['Group'].isin(['wy', 'pa', 'hr'])
    idf = vdf[vdf_bool]
    csg = Telemetry(
        times=idf['TimeUTC'],
        times_local=idf['TimeLocal'],
        lons=idf['Longitude'],
        lats=idf['Latitude'],
        zlocs=idf['Altitude'],
        animalids=idf['AnimalID'],
        regions=idf['Group'],
        df_add=idf,
        out_dir=output_dir,
    )
    # segment into tracks
    csg.ignore_data_based_on_vertical_speed(max_change=35.)
    csg.ignore_data_based_on_horizontal_speed(min_speed=0.25)
    csg.sort_df(['Group', 'AnimalID', 'TimeUTC'])
    csg.annotate_track_info(
        min_time_interval=10,
        min_time_duration=5 * 60,
        min_num_points=60,
        time_col='TimeLocal'
    )

    # annotate hrrr data
    csg.annotate_hrrr_data(tracks_only=True)
    csg.df['WindSpeed_80m'] = csg.df['WindSpeedU_10m']**2 + \
        csg.df['WindSpeedV_10m']**2
    csg.df['WindDirection_80m'] = get_wind_direction(
        csg.df['WindSpeedU_10m'],
        csg.df['WindSpeedV_10m']
    )
    csg.df.to_parquet(ann_fpath)

    # orographic updraft
    gf_func = partial(
        gaussian_filter,
        sigma=10,
        mode='constant',
        truncate=5,
        cval=0
    )
    annotate_fn = partial(
        csg.annotate_3dep_data,
        heading_col='HeadingHor_TU',
        tracks_only=True,
        filter_func=gf_func
    )

    def annotate_oro(flag):
        """annotate orographic updrafts"""
        jname = f'OroUpdraftSmooth_80m{flag}'
        wdirn = 'WindDirection_80m'
        wspeed = 'WindSpeed_80m'
        term1 = f'OroTerm1{flag}'
        term2 = f'OroTerm2{flag}'
        csg.df[jname] = np.cos(np.radians(csg.df[wdirn])) * csg.df[term1]
        csg.df[jname] += np.sin(np.radians(csg.df[wdirn])) * csg.df[term2]
        csg.df[jname] *= csg.df[wspeed]
        csg.df[f'{jname}_mod'] = expit(2 * (csg.df[jname] - 0.75))
        csg.df[jname].clip(lower=0., inplace=True)
        csg.df[f'OroUpdraft_80m{flag}'] = calcOrographicUpdraft_original(
            wspeed=csg.df[wspeed].values,
            wdirn=csg.df[wdirn].values,
            slope=csg.df[f'GroundSlope{flag}'].values,
            aspect=csg.df[f'GroundAspect{flag}'].values,
            res_terrain=10.,
            res=10.
        )
    flag_dict = {
        '': (0, 0),
        # '_d50h0': (50, 0),
        # '_d100h0': (100, 0),
        # '_d200h0': (200, 0),
        # '_d50h30l': (50, -30),
        # '_d50h30r': (50, 30),
        # '_d100h30l': (100, -30),
        # '_d100h30r': (100, 30),
    }
    for k, v in flag_dict.items():
        csg.printit(f'Annotating orographic updraft {k}..')
        annotate_fn(flag=k, dist_away=v[0], angle_away=v[1])
        annotate_oro(flag=k)

    # drop irrelavant columns
    def drop_these(istr):
        """Drop these columns"""
        csg.df.drop(
            columns=[i for i in csg.df.columns if istr in i],
            inplace=True
        )
    drop_these('Term')
    drop_these('Slope_')
    drop_these('Aspect_')
    csg.df.to_parquet(ann_fpath)

    # end
    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)
