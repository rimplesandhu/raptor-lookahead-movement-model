"""CSG Data processing script"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
import os
import time
import pickle
from functools import partial
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from bayesfilt.telemetry.utils import run_loop, add_angles
from bayesfilt.telemetry import Data3DEP, DataHRRR, Telemetry, TelemetryPlotter

warnings.filterwarnings("ignore")

fig_dir = os.path.join('/home/rsandhu/bayesfilt/examples/csg/figs')
out_dir = os.path.join('/home/rsandhu/bayesfilt/examples/csg/output')
time_lags = [5, 10, 20]


def plot_sim_tracks_in_time(ituple, colnames):
    """Plots random track"""
    list_of_simdf = None
    if len(ituple) == 3:
        idx, idf, _ = ituple
    elif len(ituple) == 4:
        idx, idf, _, list_of_simdf = ituple

    nrows = len(colnames) // 2 + len(colnames) % 2
    fig, ax = plt.subplots(nrows, 2, figsize=(10, 1.5 * nrows), sharex=True)
    ax = ax.flatten()
    for i, iname in enumerate(colnames):
        ax[i].plot(
            idf['TrackTimeElapsed'],
            idf[iname],
            '-r',
            linewidth=1.,
            alpha=0.8
        )
        if list_of_simdf is not None:
            for jdf in list_of_simdf:
                ax[i].plot(
                    jdf['TrackTimeElapsed'],
                    jdf[iname],
                    '-b',
                    linewidth=0.5,
                    alpha=0.1
                )
        ax[i].set_ylabel(iname)
        ax[i].grid(True)

    ax[-1].set_xlabel('Time elapsed [s]')
    ax[-2].set_xlabel('Time elapsed [s]')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'tracks_in_time_{str(idx)}.png'))
    return fig, ax


def plot_sim_tracks_in_space(ituple, max_agl=10000, time_pad=1):
    """Plots random track"""
    list_of_simdf = None
    if len(ituple) == 3:
        idx, idf, ids = ituple
    elif len(ituple) == 4:
        idx, idf, ids, list_of_simdf = ituple

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax = ax.flatten()
    cm = ax[0].pcolormesh(
        ids.x.values,
        ids.y.values,
        ids['GroundElevation'],
        cmap='terrain',
        alpha=0.15
    )
    cm = ax[0].scatter(
        idf['PositionX'],
        idf['PositionY'],
        c=idf['Agl'],
        s=0.4,
        cmap='Blues_r'
    )
    ax[0].plot(idf['PositionX'].iloc[0], idf['PositionY'].iloc[0], '*g')
    #ax[0].plot(idf['PositionX'].iloc[-1], idf['PositionY'].iloc[-1], '*r')
    _ = fig.colorbar(cm, ax=ax[0], label='Altitude AGL [m]')

    if list_of_simdf is not None:
        for simdf in list_of_simdf:
            ax[0].plot(
                simdf['PositionX'],
                simdf['PositionY'],
                '-r',
                alpha=0.1,
                linewidth=0.5
            )
    rnge = [[ids.x.min(), ids.x.max()], [ids.y.min(), ids.y.max()]]
    sdf = pd.concat(list_of_simdf)
    sdf = sdf[(sdf['TrackTimeElapsed'] > time_pad) & (sdf['Agl'] < max_agl)]
    H, xedges, yedges = np.histogram2d(
        sdf['PositionX'],
        sdf['PositionY'],
        bins=100,
        range=rnge
    )
    gf_function = partial(
        gaussian_filter,
        sigma=1,
        mode='constant',
        truncate=5,
        cval=0
    )
    X, Y = np.meshgrid(xedges, yedges)
    H = gf_function(H)
    H = H.T / np.amax(H)
    cm = ax[1].pcolormesh(X, Y, H, cmap='Reds', vmin=0.0)
    _ = ax[1].scatter(
        idf['PositionX'],
        idf['PositionY'],
        c=idf['Agl'],
        s=0.4,
        cmap='Blues_r'
    )
    ax[1].plot(idf['PositionX'].iloc[0], idf['PositionY'].iloc[0], '*g')
    #ax[1].plot(idf['PositionX'].iloc[-1], idf['PositionY'].iloc[-1], '*r')
    fig.colorbar(cm, ax=ax[1], label='Relative collision risk [m/s]')

    cm = ax[2].pcolormesh(
        ids.x.values,
        ids.y.values,
        ids['OroSmooth'],
        cmap='viridis',
        alpha=0.5
    )
    _ = fig.colorbar(cm, ax=ax[2], label='Orographic Updraft [m/s]')
    for iax in ax:
        # iax.axis('off')
        iax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'tracks_in_space_{str(idx)}.png'))
    return fig, ax


def gather_starting_locs(ituple, lag_cols, ntracks, scales):
    _, idf, _ = ituple
    xlocs = np.random.normal(
        loc=idf['PositionX'].iloc[0], scale=scales['PositionX'], size=ntracks)
    ylocs = np.random.normal(
        loc=idf['PositionY'].iloc[0], scale=scales['PositionY'], size=ntracks)
    zlocs = np.random.normal(
        loc=idf['Altitude'].iloc[0], scale=scales['Altitude'], size=ntracks)
    start_state = {'TrackTimeElapsed': 0.}
    xdiff = idf['PositionX'].iloc[60] - idf['PositionX'].iloc[0]
    ydiff = idf['PositionY'].iloc[60] - idf['PositionY'].iloc[0]
    start_state |= {'Heading': np.degrees(np.arctan2(xdiff, ydiff))}
    start_state |= {ix: idf[ix].iloc[:20].mean() for ix in lag_cols}
    start_state |= {f'{ix}Lag{iy}': idf[ix].iloc[0]
                    for ix in lag_cols for iy in time_lags}
    #start_state |= {f'{ix}Lag{iy}': 0. for ix in lag_cols for iy in time_lags}
    list_of_start_states = []
    for xloc, yloc, zloc in zip(xlocs, ylocs, zlocs):
        istate = start_state.copy()
        istate |= {'PositionX': xloc, 'PositionY': yloc, 'Altitude': zloc}
        list_of_start_states.append(istate)
    return list_of_start_states


def process_wind_conditions(idf, ihgt=80):
    """Add wind conditions"""
    istr = f'{str(int(ihgt))}m'
    sname = f'WindSpeed{istr}'
    dname = f'WindDirection{istr}'
    aname = f'WindRelativeAngle{istr}'
    idf[aname] = add_angles(idf['Heading'], -(180 + idf[dname]))
    iangle = np.radians(idf[aname])
    idf[f'WindSupport{istr}'] = np.cos(iangle) * idf[sname]
    idf[f'WindLateral{istr}'] = np.sin(iangle) * idf[sname]
    idf[f'WindLateral{istr}Abs'] = np.abs(idf[f'WindLateral{istr}'])
    return idf


def annotate_env_conditions(
    idf,
    ids,
    dict_of_dists: dict[str, list[int]],
    dict_of_angles: dict[str, int]
):
    """Annotate look ahead conditions"""
    if 'GroundElevation' in list(ids.data_vars.keys()):
        ids = ids.rename({'GroundElevation': 'Elev'})
    ds_loc = ids.sel(
        x=idf['PositionX'],
        y=idf['PositionY'],
        method='nearest'
    )
    for key, val in ds_loc.data_vars.items():
        idf[key] = deepcopy(val.item())
    oro_str = 'OroSmooth'
    elev_str = 'Elev'
    iangle = np.radians(idf['Heading'])
    for iname in [oro_str, elev_str]:
        for _, ilist in dict_of_dists.items():
            for idist in ilist:
                xloc = idf['PositionX'] + idist * np.sin(iangle)
                yloc = idf['PositionY'] + idist * np.cos(iangle)
                jname = f'{iname}D{str(idist)}'
                idf[jname] = ids[iname].sel(
                    x=xloc,
                    y=yloc,
                    method='nearest'
                ).item()
    iflag = 'Near'
    for ikey, ival in dict_of_angles.items():
        for idist in dict_of_dists[iflag]:
            iangle = np.radians(add_angles(idf['Heading'], -(180 + ival)))
            xloc = idf['PositionX'] + idist * np.sin(iangle)
            yloc = idf['PositionY'] + idist * np.cos(iangle)
            for iname in [oro_str, elev_str]:
                jname = f'{iname}D{str(idist)}{ikey}'
                idf[jname] = ids[iname].sel(
                    x=xloc,
                    y=yloc,
                    method='nearest'
                ).item()
    return idf


def annotate_lookahead_conditions(
    idf,
    dict_of_dists: dict[str, list[int]],
    oro_fn,
    elev_fn
):
    """Process lookahead conditions"""
    is_dict = isinstance(idf, dict)
    if is_dict:
        idf = pd.DataFrame({k: [v] for k, v in idf.items()})
    for ikey, ival in dict_of_dists.items():
        for istr, ifn in zip(['OroSmooth', 'Elev'], [oro_fn, elev_fn]):
            in_cols = [f'{istr}D{str(ix)}' for ix in ival]
            out_col = f'{istr}{ikey}'
            #print(out_col, in_cols)
            idf[out_col] = ifn(idf[in_cols].values, axis=1)
            idf[f'{out_col}Diff'] = idf[out_col] - idf[istr]
            idf.drop(columns=in_cols, inplace=True)
            if ikey == 'Near':
                for iflag in ['L30', 'R30']:
                    in_cols = [f'{istr}D{str(ix)}{iflag}' for ix in ival]
                    out_col = f'{istr}{ikey}{iflag}'
                    #print(out_col, in_cols)
                    ixx = idf[in_cols].replace(np.inf, np.nan)
                    ixx = ixx.bfill().ffill().values
                    idf[out_col] = ifn(ixx, axis=1)
                    idf[f'{out_col}Diff'] = idf[out_col] - idf[f'{istr}{ikey}']
                    idf.drop(columns=[out_col], inplace=True)
                    idf.drop(columns=in_cols, inplace=True)
    idf.drop(
        columns=[ix for ix in idf.columns if 'ElevD' in ix],
        inplace=True
    )
    idf.drop(
        columns=[ix for ix in idf.columns if 'OroSmoothD' in ix],
        inplace=True
    )
    if is_dict:
        idf = idf.to_dict('records')[0]
    return idf


def annotate_state(
    ids,
    istate
):
    """Annotate state"""
    dist_dict = {
        'Near': [25, 50, 75], 'Mid': [400, 500, 600], 'Far': [900, 1000, 1100]
    }
    angle_dict = {'L30': -30, 'R30': 30}
    istate = annotate_env_conditions(istate, ids, dist_dict, angle_dict)
    istate = process_wind_conditions(istate, 80.)
    istate = annotate_lookahead_conditions(istate, dist_dict, np.mean, np.mean)
    istate['HeadingRateAbs'] = np.abs(istate['HeadingRate'])
    istate['Agl'] = istate['Altitude'] - istate['Elev']
    return istate


def reached_the_edge(istate, width):
    is_at_boundary = (istate['PositionX'] < 0) | (istate['PositionY'] < 0)
    is_at_boundary = is_at_boundary | (
        istate['PositionY'] > width[1]) | (istate['PositionX'] > width[0])
    return is_at_boundary


def compute_updated_predictables(istate, model_df, std_functions, reg_type):
    """Computed updated predictables"""
    mean_model = reg_type
    err_model = f'{reg_type}Error'
    for icol in list(model_df.columns):
        cov_std_names = model_df.loc['Covariates', icol]
        cov_names = [ix.split('Std')[0] for ix in cov_std_names]
        std_vec = []
        for ixx, iy in zip(cov_std_names, cov_names):
            ival = np.atleast_2d(istate[iy])
            jval = std_functions[ixx].transform(ival)[0][0]
            std_vec.append(jval)
            #print(ix, iy, ival[0][0], jval)
        std_vec = np.atleast_2d(std_vec)
        std_vec_poly = model_df.loc['Features', icol].transform(std_vec)
        std_vec_poly = np.atleast_2d(std_vec_poly)
        pred_std = model_df.loc[mean_model, icol].predict(std_vec_poly)
        std_vec_poly = model_df.loc['FeaturesError', icol].transform(std_vec)
        std_vec_poly = np.atleast_2d(std_vec_poly)
        pred_std_err = model_df.loc[err_model, icol].predict(std_vec_poly)
        pred_std_err = max(0.001, pred_std_err)
        #assert pred_std_err > 0., 'Variance getting negative'
        pred_std += np.random.normal(loc=0, scale=0.25 * np.sqrt(pred_std_err))
        pred_std = np.atleast_2d(pred_std)
        cname = icol.split('Next')[0]
        istate[cname] = std_funcs[icol].inverse_transform(pred_std)[0][0]
    return istate


def generate_track(
    start_state,
    num_steps,
    ids,
    models,
    std_funcs,
    mtype,
    verbose=True
):
    """Generate track"""
    simdf = []
    cstate = start_state.copy()
    cstate = annotate_state(ids, cstate)
    simdf.append(cstate.copy())
    pstate = cstate.copy()
    width = [ids.x.max().item(), ids.y.max().item()]
    for i in range(num_steps):
        # new values for predictables
        cstate = compute_updated_predictables(cstate, models, std_funcs, mtype)
        if cstate['VelocityHor'] < 0.:
            cstate['VelocityHor'] = deepcopy(pstate['VelocityHor'])
        fc = 0.75
        hrate = fc * cstate['HeadingRate'] + (1 - fc) * pstate['HeadingRate']
        hspeed = fc * cstate['VelocityHor'] + (1 - fc) * pstate['VelocityHor']
        vspeed = fc * cstate['VelocityVer'] + (1 - fc) * pstate['VelocityVer']

        cstate['Heading'] = add_angles(cstate['Heading'], hrate)
        iangle = np.radians(cstate['Heading'])
        cstate['PositionX'] += np.sin(iangle) * hspeed
        cstate['PositionY'] += np.cos(iangle) * hspeed
        cstate['Altitude'] += vspeed
        cstate['TrackTimeElapsed'] += 1.
        cstate = annotate_state(ids, cstate)
        for ilag in time_lags:
            if cstate['TrackTimeElapsed'] >= ilag:
                for icol in list(models.columns):
                    cname = icol.split('Next')[0]
                    cname_lag = f'{cname}Lag{str(int(ilag))}'
                    cstate[cname_lag] = deepcopy(simdf[-ilag][cname])
        if verbose:
            print(f'---{i}----')
            for ikey, ival in cstate.items():
                pval = np.around(pstate[ikey], 3)
                print(f'{ikey}: {pval}->{np.around(ival,3)}')
        if reached_the_edge(cstate, width):
            break
        if (cstate['VelocityHor'] < 0.) or (abs(cstate['VelocityVer']) > 50.):
            break
        simdf.append(cstate.copy())
        pstate = cstate.copy()
    return pd.DataFrame(simdf)


if __name__ == "__main__":

    print('\n---Generate simulated tracks', flush=True)
    start_time = time.time()

    # load resampled data
    print('Loading telemetry data..', flush=True)
    output_dir = os.path.join('/home/rsandhu/projects_car/csg_data/output')
    fpath = os.path.join(output_dir, 'telemetry')
    fpath = os.path.join(fpath, 'csg_ge_vr.prq_tracks_ca_annotated')
    df = pd.read_parquet(fpath)
    df.columns = [ix.replace('OroUpdraft', 'Oro') for ix in df.columns]
    df.columns = [ix.replace('GroundElevation', 'Elev') for ix in df.columns]
    df['WindLateral80mAbs'] = df['WindLateral80m'].abs()
    df['Agl'] = df['Altitude'] - df['Elev']
    df['HeadingRateRaw'] = df['HeadingRate'].multiply(-1)
    df['HeadingRate'] = df['HeadingRateRaw'].rolling(
        3, min_periods=1, center=True).mean().bfill().ffill()
    df['WindLateral80mAbs'] = df['WindLateral80m'].abs()
    df.drop(columns=[ix for ix in df.columns if '_var' in ix], inplace=True)
    df.drop(columns=[ix for ix in df.columns if '10m' in ix], inplace=True)
    df.drop(
        columns=[ix for ix in df.columns if 'OroUpdraftD' in ix], inplace=True)
    csg = Telemetry(
        times=df['TimeUTC'],
        times_local=df['TimeLocal'],
        lons=df['Longitude'],
        lats=df['Latitude'],
        regions=df['Group'],
        df_add=df,
        out_dir=output_dir
    )
    with open(csg.domain_fpath, 'rb') as outp:
        csg.df_subdomains = pickle.load(outp)
    csg_plotter = TelemetryPlotter(csg)

    # gather random set of tracks
    num_cases = 36
    time_len_sec = 180
    num_sim_tracks = 50
    print('Gathering random tracks..')
    gf_func = partial(
        gaussian_filter,
        sigma=5,
        mode='constant',
        truncate=5,
        cval=0
    )
    list_of_rnd_tracks = []
    for ix in range(num_cases):
        rdf, rds = csg.get_random_track_segment(
            time_len=time_len_sec,
            out_dir=os.path.join(out_dir, f'track_{str(ix)}'),
            rnd_seed=ix,
            gf_func=gf_func,
            max_agl=200,
            min_xwidth_km=3,
            min_ywidth_km=3,
            xpad_km=1.5,
            ypad_km=1.5
        )
        list_of_rnd_tracks.append((ix, rdf, rds))

    # plotting random tracks
    _ = run_loop(plot_sim_tracks_in_space, list_of_rnd_tracks)

    # gather models
    fpath = os.path.join(output_dir, 'fitted_models_ca.pickle')
    with open(fpath, 'rb') as outp:
        models = pickle.load(outp)
    fpath = os.path.join(output_dir, 'fitted_models_ca_std_funcs.pickle')
    with open(fpath, 'rb') as outp:
        std_funcs = pickle.load(outp)

    # gather starting points for each track
    print('Gathering starting locations..')
    lag_colnames = [ix.split('Next')[0] for ix in models.columns]
    fn = partial(
        gather_starting_locs,
        lag_cols=lag_colnames,
        ntracks=num_sim_tracks,
        scales={'PositionX': 50, 'PositionY': 50, 'Altitude': 5.}
    )
    list_of_startlocs = run_loop(fn, list_of_rnd_tracks)
    zipper = zip(list_of_rnd_tracks, list_of_startlocs)
    list_of_inputs = [ix + (iy,) for ix, iy in zipper]
    with open(os.path.join(output_dir, 'sim_results.pickle'), "wb") as f:
        pickle.dump(list_of_rnd_tracks, f)

    # generating tracks
    print('Simulating tracks..')

    def generate_alltracks(ituple):
        idx, idf, ids, start_states = ituple
        fn = partial(
            generate_track,
            ids=ids,
            models=models,
            std_funcs=std_funcs,
            num_steps=int(idf['TrackTimeElapsed'].iloc[-1]),
            mtype='LinReg',
            verbose=False
        )
        list_of_simdf = run_loop(fn, start_states, ncores=1, desc=str(idx))
        # for start_state in start_states:
        #     sdf = generate_track(
        #         start_state=start_state,
        #         ids=ids,
        #         models=models,
        #         std_funcs=std_funcs,
        #         num_steps=int(idf['TrackTimeElapsed'].iloc[-1]),
        #         mtype='LinReg',
        #         verbose=False
        #     )
        #     list_of_simdf.append(sdf)
        return list_of_simdf

    out_list = run_loop(generate_alltracks, list_of_inputs)
    zipper = zip(list_of_rnd_tracks, out_list)
    list_of_outs = [ix + (iy,) for ix, iy in zipper]
    with open(os.path.join(output_dir, 'sim_results.pickle'), "wb") as f:
        pickle.dump(list_of_outs, f)

    run_time = np.around(((time.time() - start_time)) / 60., 2)
    print(f'---took {run_time} mins\n', flush=True)

    #  for ihgt in list_of_heights:
    #         istr = f'{str(int(ihgt))}m'
    #         speed_col = f'WindSpeed{istr}'
    #         dirn_col = f'WindDirection{istr}'
    #         angle_col = f'WindRelativeAngle{istr}'
    #         self.df[speed_col] = np.sqrt(
    #             self.df[f'WindSpeedU{istr}']**2 +
    #             self.df[f'WindSpeedV{istr}']**2
    #         )
    #         self.df[dirn_col] = get_wind_direction(
    #             self.df[f'WindSpeedU{istr}'],
    #             self.df[f'WindSpeedV{istr}']
    #         )
    #         self.df[angle_col] = self.df[self.heading_col].subtract(
    #             180 + self.df[dirn_col])
    #         self.df.loc[self.df[angle_col] > 180, angle_col] -= 360
    #         self.df.loc[self.df[angle_col] < -180, angle_col] += 360
    #         self.df[f'WindSupport{istr}'] = np.cos(np.radians(
    #             self.df[angle_col])) * self.df[speed_col]
    #         self.df[f'WindLateral{istr}'] = np.sin(np.radians(
    #             self.df[angle_col])) * self.df[speed_col]
