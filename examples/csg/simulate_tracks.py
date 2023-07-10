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

CSG_DIR = os.path.join('/home/rsandhu/projects_car/csg_data')
OUT_DIR = os.path.join(CSG_DIR, 'output')
TELEMETRY_DIR = os.path.join(OUT_DIR, 'telemetry')
TRACK_DIR = os.path.join(OUT_DIR, 'tracks')
BF_CSG_DIR = os.path.join('/home/rsandhu/bayesfilt/examples/csg')
FIG_DIR = os.path.join(BF_CSG_DIR, 'figs')
clrs = ['#377eb8', '#ff7f00', '#4daf4a',
        '#f781bf', '#a65628', '#984ea3',
        '#999999', '#e41a1c', '#dede00']
time_lags = [5, 10, 30, 60, 120]
dist_dict = {
    'Fifty': [40, 50, 60],
    'Onehd': [80, 100, 120],
    'Twohd': [200, 250, 300],
    'Fvehd': [400, 500, 600],
    'Onekm': [800, 1000, 1200],
    'Twokm': [1600, 2000, 2400],
}
angle_dict = {
    'L15': -15, 'R15': 15,
    'L30': -30, 'R30': 30,
    'L60': -60, 'R60': 60
}
angle_cases = ['Fifty', 'Onehd', 'Twohd']


def get_annotated_telemetry_df():
    fpath = os.path.join(
        TELEMETRY_DIR, 'csg_ge_vr.prq_tracks_ca_annotated2')
    df = pd.read_parquet(fpath)
    df['Agl'] = df['Altitude'] - df['GroundElevation']
    df['HeadingRateRaw'] = df['HeadingRate'].multiply(-1)
    df['HeadingRate'] = df['HeadingRateRaw'].rolling(
        3, min_periods=1, center=True).mean().bfill().ffill()
    df['HeadingRateAbs'] = df['HeadingRate'].abs()
    df['WindLateral80mAbs'] = df['WindLateral80m'].abs()
    df['HrateByHspeed'] = df['HeadingRate'].divide(
        df['VelocityHor'].clip(lower=0.5, upper=np.inf))
    df['VelocityHor_var'] = (df['VelocityX_var'] + df['VelocityY_var']) / 2.
    # df.drop(columns=[ix for ix in df.columns if '_var' in ix], inplace=True)
    # df.drop(columns=[ix for ix in df.columns if '10m' in ix], inplace=True)
    # df.drop(columns=[ix for ix in df.columns if 'OroD' in ix], inplace=True)
    df.columns = [ix.replace('GroundElevation', 'Elev') for ix in df.columns]
    df.columns = [ix.replace('OroUpdraft', 'Oro') for ix in df.columns]
    # df = annotate_lookahead_conditions(df, oro_fn=np.mean, elev_fn=np.mean)
    return df


def get_short_df(idf, num_samples):
    """Get short df for calibration"""
    # select specific data
    ibool = (idf['Group'].isin(['pa', 'wy', 'hr']))

    # trim to validate the lag terms
    ibool = ibool & (idf['TrackTimeElapsed'] > max(time_lags))

    # trim based on agl and elev
    ibool = ibool & (idf['Agl'] < 200) & (idf['Agl'] > -50)

    # # elev diff
    # rnge = (-400, 400)
    # for icase in list(dist_dict.keys()):
    #     icol = f'Elev{icase}Diff'
    #     ibool = ibool & (idf[icol] > rnge[0]) & (idf[icol] < rnge[1])

    # # oro
    # rnge = (-10, 10.)
    # for icase in list(dist_dict.keys()):
    #     icol = f'OroSmooth{icase}Diff'
    #     ibool = ibool & (idf[icol] > rnge[0]) & (idf[icol] < rnge[1])

    # oro
    # rnge = (-1, 6.)
    # for icase in list(dist_dict.keys()):
    #     icol = f'OroSmooth{icase}'
    #     ibool = ibool & (idf[icol] > rnge[0]) & (idf[icol] < rnge[1])

    # trim base don heading rate
    ibool = ibool & (idf['HeadingRate'] < 30.) & (idf['HeadingRate'] > -30.)
    ibool = ibool & (idf['HeadingRateNext'] < 30) & (
        idf['HeadingRateNext'] > -30)
    lim_val = 30
    for ilag in time_lags:
        vname = f'HeadingRateLag{ilag}'
        ibool = ibool & (idf[vname] < lim_val) & (idf[vname] > -lim_val)

    # # trim base don heading rate
    # ibool = ibool & (df['HrateByHspeed']<10.) &  (df['HeadingRate']>-10.)
    # ibool = ibool & (df['HrateByHspeedNext']<10) &  (df['HrateByHspeedNext']>-10)
    # lim_val = 10
    # for ilag in time_lags:
    #     vname = f'HrateByHspeedLag{ilag}'
    #     ibool = ibool & (df[vname]<lim_val) &  (df[vname]>-lim_val)

    # trim based on hor speed
    ibool = ibool & (idf['VelocityHor'] < 25.) & (idf['VelocityHor'] > 1.)
    lim_val = 30
    for ilag in time_lags:
        vname = f'VelocityHorLag{ilag}'
        ibool = ibool & (idf[vname] < lim_val) & (idf[vname] > -lim_val)

    # trim based on hor speed
    ibool = ibool & (idf['VelocityVer'] < 5.) & (idf['VelocityVer'] > -5)
    lim_val = 5
    for ilag in time_lags:
        vname = f'VelocityVerLag{ilag}'
        ibool = ibool & (idf[vname] < lim_val) & (idf[vname] > -lim_val)

    # get the training data
    ibool = ibool & (idf.isnull().sum(axis=1) == 0)
    dfshort = idf[ibool].sample(
        n=min(num_samples, ibool.sum()-1), axis=0).copy()
    # print(ibool.sum()*100/df.shape[0], dfshort.shape[0]*100/df.shape[0])
    # print(ibool.sum(), dfshort.shape[0], df.shape[0])
    # dfshort['AglLog'] = dfshort['Agl'].add(51).divide(100.).apply(np.log)
    # cols_to_mod = [
    #     'WindLateral80m', 'OroSmoothNearL30Diff', 'OroSmoothNearR30Diff',
    #     'OroSmoothCloseL30Diff', 'OroSmoothCloseR30Diff',
    # ]
    # for icol in cols_to_mod:
    #     dfshort[f'{icol}Mod'] = dfshort[icol]*np.sign(dfshort['HeadingRate'])
    return dfshort


def compute_predictables(idf):
    """COmpute predictables"""
    variables = {}
    variables['Pred'] = ['VelocityHor', 'HeadingRate', 'VelocityVer']
    # variables['Pred'] = ['VelocityHor', 'HrateByHspeed', 'VelocityVer']
    # variables['PredChange'] = [f'{ix}Change' for ix in variables['Pred']]
    variables['PredNext'] = [f'{ix}Next' for ix in variables['Pred']]
    for i, ivar in enumerate(variables['Pred']):
        # df[variables['PredChange'][i]] = df[ivar].diff(-1).ffill().bfill()
        idf[variables['PredNext'][i]] = idf[ivar].shift(-1).ffill().bfill()
    for ilag in time_lags:
        variables[f'PredLag{ilag}'] = [
            f'{ix}Lag{ilag}' for ix in variables['Pred']]
        for ix, iy in zip(variables[f'PredLag{ilag}'], variables['Pred']):
            idf[ix] = idf[iy].shift(ilag).ffill().bfill()
    return idf, variables


def plot_sim_tracks_in_time(ituple, colnames, savefig=False):
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
                if iname == 'Agl':
                    idata = jdf[iname] - \
                        jdf[iname].iloc[0] + idf[iname].iloc[0]
                else:
                    idata = jdf[iname]
                ax[i].plot(
                    jdf['TrackTimeElapsed'],
                    idata,
                    '-b',
                    linewidth=0.5,
                    alpha=0.1
                )
        ax[i].set_ylabel(iname)
        ax[i].grid(True)

    ax[-1].set_xlabel('Time elapsed [s]')
    ax[-2].set_xlabel('Time elapsed [s]')
    fig.tight_layout()
    if savefig:
        plt.savefig(os.path.join(FIG_DIR, f'tracks_in_time_{str(idx)}.png'))
    return fig, ax


def plot_sim_tracks_in_space2(
        ituple,
        max_agl=10000,
        time_pad=[30, 60],
        zoom=100,
        savefig=False
):
    """Plots random track"""
    list_of_simdf = None
    if len(ituple) == 3:
        idx, idf, ids = ituple
    elif len(ituple) == 4:
        idx, idf, ids, list_of_simdf = ituple

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.75),
                           gridspec_kw={'width_ratios': [1.25, 1]})
    ax = ax.flatten()
    ax[0].plot(idf['PositionX'].iloc[0],
               idf['PositionY'].iloc[0], '*g', label='Start')
    ndf = idf[idf['TrackTimeElapsed'].between(*time_pad)]
    ax[0].plot(ndf['PositionX'], ndf['PositionY'], '-r',
               linewidth=2, alpha=0.75,
               label='Telemetry')
    ax[0].plot(idf['PositionX'], idf['PositionY'], '-r',
               linewidth=1, alpha=0.2)
    etime = idf['TrackTimeElapsed'].values
    angle = np.radians(idf['Heading'].iloc[:10].values)
    hspeed = idf['VelocityHor'].iloc[:10].mean()
    xlocs = idf['PositionX'].iloc[0] + np.mean(np.sin(angle)) * hspeed * etime
    ylocs = idf['PositionY'].iloc[0] + np.mean(np.cos(angle)) * hspeed * etime
    # ax[0].plot(np.mean(xlocs), np.mean(ylocs), 'ok')
    ibool = (etime > time_pad[0]) & (etime < time_pad[1])
    ax[0].plot(xlocs[ibool], ylocs[ibool], '-k',
               linewidth=2., alpha=0.75, label='CV')
    ax[0].plot(xlocs[~ibool], ylocs[~ibool], '-k',
               linewidth=1., alpha=0.2)
    ax[0].legend(loc=2, bbox_to_anchor=(0, 1.125), borderaxespad=0.,
                 ncol=3, columnspacing=1, handletextpad=0.2, handlelength=1.,
                 frameon=False)
    rnge = [[ids.x.min(), ids.x.max()], [ids.y.min(), ids.y.max()]]
    gf_function = partial(
        gaussian_filter,
        sigma=1,
        mode='constant',
        truncate=5,
        cval=0
    )

    if list_of_simdf is not None:
        # for simdf in list_of_simdf:
        #     ax[0].plot(
        #         simdf['PositionX'],
        #         simdf['PositionY'],
        #         '-b',
        #         alpha=0.1,
        #         linewidth=0.5
        #     )
        sdf = pd.concat(list_of_simdf)
        sdf = sdf[(sdf['TrackTimeElapsed'].between(*time_pad))
                  & (sdf['Agl'] < max_agl)]
        H, xedges, yedges = np.histogram2d(
            sdf['PositionX'],
            sdf['PositionY'],
            bins=100,
            range=rnge
        )
        X, Y = np.meshgrid(xedges, yedges)
        H = gf_function(H)
        H = H.T / np.amax(H)
        cm = ax[0].pcolormesh(X, Y, H, cmap='Blues', vmin=0.0, alpha=0.75)
        fig.colorbar(
            cm, ax=ax[0], label='Relative collision risk', pad=0.01)

    # cm = ax[0].scatter(
    #     ndf['PositionX'],
    #     ndf['PositionY'],
    #     c=ndf['Agl'],
    #     s=0.4,
    #     cmap='autumn',
    #     vmin=0,
    #     #vmax=np.ceil(np.amax(idf['Agl']) / 25.) * 25.,
    #     vmax=200
    # )
    # _ = fig.colorbar(cm, ax=ax[0], label='Altitude AGL [m]', pad=0.01)
    cm = ax[1].pcolormesh(
        ids.x.values,
        ids.y.values,
        ids['OroSmooth'],
        cmap='cividis',
        alpha=0.5,
        vmin=0,
        vmax=2.
    )
    _ = fig.colorbar(cm, ax=ax[1], label='Orographic Updraft [m/s]', pad=0.01)
    ax[1].plot(idf['PositionX'].iloc[0],
               idf['PositionY'].iloc[0], '*g', label='Start')

    for iax in ax:
        iax.axis('off')
        iax.set_aspect('equal')
        iax.set_xlim([ids.x.mean() - zoom, ids.x.mean() + zoom])
        iax.set_ylim([ids.y.mean() - zoom, ids.y.mean() + zoom])
    fig.tight_layout()
    if savefig:
        plt.savefig(os.path.join(
            FIG_DIR, f'fig_{str(idx)}.png'), bbox_inches='tight')
    return fig, ax


def plot_sim_tracks_in_space(ituple, max_agl=10000, time_pad=1, savefig=False):
    """Plots random track"""
    list_of_simdf = None
    if len(ituple) == 3:
        idx, idf, ids = ituple
    elif len(ituple) == 4:
        idx, idf, ids, list_of_simdf = ituple

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
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
    # ax[0].plot(idf['PositionX'].iloc[-1], idf['PositionY'].iloc[-1], '*r')
    _ = fig.colorbar(cm, ax=ax[0], label='Altitude AGL [m]')
    rnge = [[ids.x.min(), ids.x.max()], [ids.y.min(), ids.y.max()]]
    gf_function = partial(
        gaussian_filter,
        sigma=1,
        mode='constant',
        truncate=5,
        cval=0
    )

    if list_of_simdf is not None:
        for simdf in list_of_simdf:
            ax[0].plot(
                simdf['PositionX'],
                simdf['PositionY'],
                '-r',
                alpha=0.1,
                linewidth=0.5
            )
        sdf = pd.concat(list_of_simdf)
        sdf = sdf[(sdf['TrackTimeElapsed'] > time_pad)
                  & (sdf['Agl'] < max_agl)]
        H, xedges, yedges = np.histogram2d(
            sdf['PositionX'],
            sdf['PositionY'],
            bins=100,
            range=rnge
        )
        X, Y = np.meshgrid(xedges, yedges)
        H = gf_function(H)
        H = H.T / np.amax(H)
        cm = ax[1].pcolormesh(X, Y, H, cmap='Reds', vmin=0.0)
        fig.colorbar(cm, ax=ax[1], label='Relative collision risk [m/s]')

    _ = ax[1].scatter(
        idf['PositionX'],
        idf['PositionY'],
        c=idf['Agl'],
        s=0.4,
        cmap='Blues_r'
    )
    ax[1].plot(idf['PositionX'].iloc[0], idf['PositionY'].iloc[0], '*g')
    # ax[1].plot(idf['PositionX'].iloc[-1], idf['PositionY'].iloc[-1], '*r')

    cm = ax[2].pcolormesh(
        ids.x.values,
        ids.y.values,
        ids['OroSmooth'],
        cmap='viridis',
        alpha=0.5
    )
    _ = fig.colorbar(cm, ax=ax[2], label='Orographic Updraft [m/s]')
    for iax in ax:
        iax.axis('off')
        iax.set_aspect('equal')
        iax.set_xlim([ids.x.min(), ids.x.max()])
        iax.set_ylim([ids.y.min(), ids.y.max()])
    fig.tight_layout()
    if savefig:
        plt.savefig(os.path.join(FIG_DIR, f'tracks_in_space_{str(idx)}.png'))
    return fig, ax


def gather_starting_locs(ituple, ntracks):
    _, idf, _ = ituple
    # use past data
    xdiff = idf['PositionX'].iloc[10] - idf['PositionX'].iloc[0]
    ydiff = idf['PositionY'].iloc[10] - idf['PositionY'].iloc[0]
    heading = np.degrees(np.arctan2(xdiff, ydiff))
    list_of_start_states = [{'TrackTimeElapsed': 0.} for _ in range(ntracks)]
    var_cols = [
        'PositionX', 'PositionY', 'Altitude', 'VelocityVer', 'VelocityHor'
    ]
    lag_cols = [
        'VelocityHor', 'VelocityVer', 'HeadingRate'
    ]
    for i, istate in enumerate(list_of_start_states):
        for icol in var_cols:
            istate[icol] = np.random.normal(
                loc=idf[icol].iloc[0],
                scale=np.sqrt(idf[f'{icol}_var'].iloc[0])
            )
        istate['Heading'] = add_angles(np.random.normal(
            loc=heading,
            scale=2.
        ), 0)
        istate['HeadingRate'] = np.random.normal(
            loc=idf['HeadingRate'].iloc[0],
            scale=0.1
        )
        istate['VelocityHor'] = abs(istate['VelocityHor'])
        for ikey in lag_cols:
            istate |= {f'{ikey}Lag{iy}': np.random.rand() *
                       istate[ikey] for iy in time_lags}
            # sample it from latest val
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
        for _, ilist in dist_dict.items():
            for idist in ilist:
                xloc = idf['PositionX'] + idist * np.sin(iangle)
                yloc = idf['PositionY'] + idist * np.cos(iangle)
                jname = f'{iname}D{str(idist)}'
                idf[jname] = ids[iname].sel(
                    x=xloc,
                    y=yloc,
                    method='nearest'
                ).item()
    for ikey, ival in angle_dict.items():
        for iflag in angle_cases:
            for idist in dist_dict[iflag]:
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
    oro_fn,
    elev_fn
):
    """Process lookahead conditions"""
    is_dict = isinstance(idf, dict)
    if is_dict:
        idf = pd.DataFrame({k: [v] for k, v in idf.items()})
    for ikey, ival in dist_dict.items():
        for istr, ifn in zip(['OroSmooth', 'Elev'], [oro_fn, elev_fn]):
            in_cols = [f'{istr}D{str(ix)}' for ix in ival]
            out_col = f'{istr}{ikey}'
            # print(out_col, in_cols)
            idf[out_col] = ifn(idf[in_cols].values, axis=1)
            idf[f'{out_col}Diff'] = idf[out_col] - idf[istr]
            idf.drop(columns=in_cols, inplace=True)
            if ikey in angle_cases:
                for iflag, _ in angle_dict.items():
                    in_cols = [f'{istr}D{str(ix)}{iflag}' for ix in ival]
                    out_col = f'{istr}{ikey}{iflag}'
                    # print(out_col, in_cols)
                    ixx = idf[in_cols].replace(np.inf, np.nan)
                    ixx = ixx.bfill().ffill().values
                    idf[out_col] = ifn(ixx, axis=1)
                    idf[f'{out_col}Diff'] = idf[out_col] - idf[f'{istr}{ikey}']
                    # idf[f'{out_col}Diff'] = idf[out_col] - idf[f'{istr}']
                    idf.drop(columns=[out_col], inplace=True)
                    idf.drop(columns=in_cols, inplace=True)
    idf.drop(
        columns=[ix for ix in idf.columns if 'ElevD' in ix],
        inplace=True
    )
    # idf.drop(
    #     columns=[ix for ix in idf.columns if 'OroSmoothD' in ix],
    #     inplace=True
    # )
    if is_dict:
        idf = idf.to_dict('records')[0]
    return idf


def annotate_state(
    ids,
    istate
):
    """Annotate state"""

    istate = annotate_env_conditions(istate, ids)
    istate = process_wind_conditions(istate, 80.)
    istate = annotate_lookahead_conditions(istate, np.mean, np.mean)
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
            # print(ix, iy, ival[0][0], jval)
        std_vec = np.atleast_2d(std_vec)
        std_vec_poly = model_df.loc['Features', icol].transform(std_vec)
        std_vec_poly = np.atleast_2d(std_vec_poly)
        pred_std = model_df.loc[mean_model, icol].predict(std_vec_poly)
        std_vec_poly = model_df.loc['FeaturesError', icol].transform(std_vec)
        std_vec_poly = np.atleast_2d(std_vec_poly)
        pred_std_err = model_df.loc[err_model, icol].predict(std_vec_poly)
        pred_std_err = max(0.001, pred_std_err)
        # assert pred_std_err > 0., 'Variance getting negative'
        pred_std += np.random.normal(loc=0, scale=0.75 * np.sqrt(pred_std_err))
        pred_std = np.atleast_2d(pred_std)
        cname = icol.split('Next')[0]
        istate[cname] = std_functions[icol].inverse_transform(pred_std)[0][0]
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
        if abs(cstate['HeadingRate']) > 40:
            cstate['HeadingRate'] = 0.5 * pstate['HeadingRate']
        if cstate['VelocityHor'] < 0.:
            cstate['VelocityHor'] = deepcopy(pstate['VelocityHor'])
        fc = 0.75
        hspeed = fc * cstate['VelocityHor'] + (1 - fc) * pstate['VelocityHor']
        vspeed = fc * cstate['VelocityVer'] + (1 - fc) * pstate['VelocityVer']
        # hrate = fc * cstate['HeadingRate'] + (1 - fc) * pstate['HeadingRate']
        # hrate = cstate['HrateByHspeed'] * hspeed
        hrate = cstate['HeadingRate']
        cstate['HeadingRate'] = fc * hrate + (1 - fc) * pstate['HeadingRate']

        cstate['Heading'] = add_angles(
            cstate['Heading'], cstate['HeadingRate'])
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
    df = get_annotated_telemetry_df()
    csg = Telemetry(
        times=df['TimeUTC'],
        times_local=df['TimeLocal'],
        lons=df['Longitude'],
        lats=df['Latitude'],
        regions=df['Group'],
        df_add=df,
        out_dir=OUT_DIR
    )
    with open(csg.domain_fpath, 'rb') as outp:
        csg.df_subdomains = pickle.load(outp)
    csg_plotter = TelemetryPlotter(csg)

    # gather random set of tracks
    num_cases = 200  # int(36 * 1)
    time_len_sec = 300
    num_sim_tracks = 100
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
            out_dir=os.path.join(TRACK_DIR, f'track_{str(ix)}'),
            rnd_seed=ix,
            gf_func=gf_func,
            max_agl=200,
            min_xwidth_km=5,
            min_ywidth_km=5,
            xpad_km=1.5,
            ypad_km=1.5
        )
        list_of_rnd_tracks.append((ix, rdf, rds))

    # plotting random tracks
    _ = run_loop(plot_sim_tracks_in_space, list_of_rnd_tracks)

    # gather models
    model_str = 'fit_ca.pickle_robust'
    with open(os.path.join(OUT_DIR, f'{model_str}_model'), 'rb') as outp:
        models = pickle.load(outp)
    with open(os.path.join(OUT_DIR, f'{model_str}_funcs'), 'rb') as outp:
        std_funcs = pickle.load(outp)
    # with open(os.path.join(OUT_DIR,f'{model_str}_df'), 'rb') as outp:
    #     dfshort = pickle.load(outp)

    # gather starting points for each track
    print('Gathering starting locations..')
    lag_colnames = [ix.split('Next')[0] for ix in models.columns]
    fn = partial(
        gather_starting_locs,
        ntracks=num_sim_tracks
    )
    list_of_startlocs = run_loop(fn, list_of_rnd_tracks)
    zipper = zip(list_of_rnd_tracks, list_of_startlocs)
    list_of_inputs = [ix + (iy,) for ix, iy in zipper]
    fpath = os.path.join(OUT_DIR, f'{model_str}_simulation2_5min_r75_100')
    with open(fpath, "wb") as f:
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
    with open(fpath, "wb") as f:
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
