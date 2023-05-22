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
    ann_fpath = os.path.join(output_dir, 'telemetry', f'{FNAME}_annotated')
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
        truncate=5,
        cval=0
    )
    annotate_fn = partial(
        csg.annotate_3dep_data,
        heading_col='Heading',
        tracks_only=False,
        filter_func=gf_func
    )

    flag_dict = {
        '': (0, 0),

        'D25': (25, 0),
        'D25L30': (25, -30),
        'D25R30': (25, 30),

        'D50': (50, 0),
        'D50L30': (50, -30),
        'D50R30': (50, 30),

        'D75': (75, 0),
        'D75L30': (75, -30),
        'D75R30': (75, 30),

        'D400': (400, 0),
        'D500': (500, 0),
        'D600': (600, 0),

        'D900': (900, 0),
        'D1000': (1000, 0),
        'D1100': (1100, 0),

        'D1800': (1800, 0),
        'D2000': (2000, 0),
        'D2200': (2200, 0),

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
