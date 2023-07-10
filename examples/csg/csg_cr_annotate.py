"""CSG Data processing script"""

import os
import time
from functools import partial
import pandas as pd
import numpy as np
import xarray as xr
import pickle
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
    FNAME = 'csg_ge_vr.prq_tracks_ca'
    vrate_fpath = os.path.join(output_dir, 'telemetry', FNAME)
    ann_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_annotated2')
    vdf = pd.read_parquet(vrate_fpath, engine='fastparquet')

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

    # annotate hrrr data
    csg.annotate_hrrr_data(tracks_only=True)
    csg.annotate_wind_conditions(list_of_heights=[10, 80])
    csg.sort_df()
    csg.df.to_parquet(ann_fpath)

    # orographic updraft
    gf_func = partial(
        gaussian_filter,
        sigma=5,
        mode='constant',
        truncate=10,
        cval=0
    )
    annotate_fn = partial(
        csg.annotate_3dep_data,
        heading_col='Heading',
        tracks_only=False,
        filter_func=gf_func
    )

# 20 %
    flag_dict = {
        '': (0, 0),

        'D40': (40, 0),
        'D40L15': (40, -15),
        'D40R15': (40, 15),
        'D40L30': (40, -30),
        'D40R30': (40, 30),
        'D40L60': (40, -60),
        'D40R60': (40, 60),

        'D50': (50, 0),
        'D50L15': (50, -15),
        'D50R15': (50, 15),
        'D50L30': (50, -30),
        'D50R30': (50, 30),
        'D50L60': (50, -60),
        'D50R60': (50, 60),

        'D60': (60, 0),
        'D60L15': (60, -15),
        'D60R15': (60, 15),
        'D60L30': (60, -30),
        'D60R30': (60, 30),
        'D60L60': (60, -60),
        'D60R60': (60, 60),

        'D80': (80, 0),
        'D80L15': (80, -15),
        'D80R15': (80, 15),
        'D80L30': (80, -30),
        'D80R30': (80, 30),
        'D80L60': (80, -60),
        'D80R60': (80, 60),

        'D100': (100, 0),
        'D100L15': (100, -15),
        'D100R15': (100, 15),
        'D100L30': (100, -30),
        'D100R30': (100, 30),
        'D100L60': (100, -60),
        'D100R60': (100, 60),

        'D120': (120, 0),
        'D120L15': (120, -15),
        'D120R15': (120, 15),
        'D120L30': (120, -30),
        'D120R30': (120, 30),
        'D120L60': (120, -60),
        'D120R60': (120, 60),


        'D200': (200, 0),
        'D200L15': (200, -15),
        'D200R15': (200, 15),
        'D200L30': (200, -30),
        'D200R30': (200, 30),
        'D200L60': (200, -60),
        'D200R60': (200, 60),

        'D250': (250, 0),
        'D250L15': (250, -15),
        'D250R15': (250, 15),
        'D250L30': (250, -30),
        'D250R30': (250, 30),
        'D250L60': (250, -60),
        'D250R60': (250, 60),

        'D300': (300, 0),
        'D300L15': (300, -15),
        'D300R15': (300, 15),
        'D300L30': (300, -30),
        'D300R30': (300, 30),
        'D300L60': (300, -60),
        'D300R60': (300, 60),

        'D400': (400, 0),
        'D500': (500, 0),
        'D600': (600, 0),

        'D800':  (800, 0),
        'D1000': (1000, 0),
        'D1200': (1200, 0),

        'D1600': (1600, 0),
        'D2000': (2000, 0),
        'D2400': (2400, 0),

    }
    with open(csg.domain_fpath, 'rb') as outp:
        csg.df_subdomains = pickle.load(outp)

    def drop_these(istr):
        """Drop these columns"""
        csg.df.drop(
            columns=[i for i in csg.df.columns if istr in i],
            inplace=True
        )

    csg.df_subdomains = pd.read_pickle(csg.domain_fpath)
    for k, v in flag_dict.items():
        csg.printit(f'Annotating orographic updraft {k}..')
        annotate_fn(flag=k, dist_away=v[0], angle_away=v[1])
        csg.annotate_orographic_updraft(flag=k)
        drop_these('OroTerm')
        drop_these('SlopeD')
        drop_these('AspectD')
        csg.df.to_parquet(ann_fpath)

    # other derived variables
    csg.df['AltitudeAgl'] = csg.df['Altitude'] - csg.df['GroundElevation']
    csg.sort_df()
    csg.df.to_parquet(ann_fpath)

    # end
    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)
