import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from bayesfilt.ipc import *
from bayesfilt.models import *
from bayesfilt.filters import *


def run_sim(
    time_tuple,
    sensor_dfs,
    obs_models,
    frame_fns,
    output_dir,
    time_res,
    fusion_engine
):
    ix, min_etime, max_etime = time_tuple
    rdf1, rdf2 = sensor_dfs
    radar_cols = ['PositionX', 'PositionY',
                  'HeadingRad', 'Velocity', 'Width', 'Length']
    ebool1 = rdf1.TimeElapsed.between(min_etime, max_etime)
    object_list_1 = pd.DataFrame({
        'SensorID': 'Radar 1',
        'TimeElapsed': rdf1.loc[ebool1, 'TimeElapsed'],
        'Data': rdf1.loc[ebool1, radar_cols].values.tolist()
    })
    ebool2 = rdf2.TimeElapsed.between(min_etime, max_etime)
    object_list_2 = pd.DataFrame({
        'SensorID': 'Radar 2',
        'TimeElapsed': rdf2.loc[ebool2, 'TimeElapsed'],
        'Data': rdf2.loc[ebool2, radar_cols].values.tolist()
    })
    object_list = pd.concat([object_list_1, object_list_2])
    object_list.sort_values(by='TimeElapsed', inplace=True)
    object_list.reset_index(inplace=True, drop=True)
    # print(object_list.head())

    fusion_engine.run(
        object_list=object_list,
        obs_models=obs_models
    )
    fdf = fusion_engine.df
    if not fdf.empty:
        fdf['TrackQuality'] = 1. / \
            np.sqrt(fdf['PositionX_var'] + fdf['PositionY_var'])
        fdf['TrackQuality'] = fdf['TrackQuality'] / fdf['TrackQuality'].max()
        fdf['TrackQuality'] = fdf['TrackQuality'].clip(lower=0., upper=1.)
        fdf['Heading'] = np.degrees(fdf['Heading'])
        # fdf['Heading'] = np.degrees(np.arctan2(
        #     fdf['VelocityX'], fdf['VelocityY']))
        fdf.rename(columns={'ObjectId': 'ObjectID'}, inplace=True)

        nframes = int((max_etime-min_etime)/time_res) + 1
        list_of_times = np.linspace(min_etime, max_etime, nframes)
        run_dir = os.path.join(output_dir, f'run{ix}')
        rdir = os.path.join(run_dir, 'radars')
        frame_fns[0](
            list_of_dfs=[rdf1[ebool1], rdf2[ebool2]],
            list_of_etimes=list_of_times,
            image_dir=rdir,
        )
        fdir = os.path.join(run_dir, 'fused')
        frame_fns[1](
            list_of_dfs=[fdf],
            list_of_etimes=list_of_times,
            image_dir=fdir,
        )
        out_dir = os.path.join(run_dir, f'merged')
        merge_frames(rdir, fdir, out_dir)
        create_video_from_images(out_dir, fps=2, filename=f'run{ix}.avi')
    return fdf


if __name__ == "__main__":

    # directories
    fig_dir = os.path.join(os.path.curdir, 'figs')
    save_dir = os.path.join(os.path.curdir, 'output')
    base_dir = '/lustre/eaglefs/projects/aumc/rsandhu/camera_radar_28aug2023/output'

    # load raw data, transformed
    print('Loading raw sensor data..', flush=True)
    rdf1 = pd.read_pickle(os.path.join(base_dir, 'radar_1_tr.pkl'))
    rdf2 = pd.read_pickle(os.path.join(base_dir, 'radar_2_tr.pkl'))
    for idf in [rdf1, rdf2]:
        idf['HeadingRad'] = np.radians(idf['Heading'])
    with open(os.path.join(base_dir, 'intersection.pkl'), 'rb') as outp:
        cs = pickle.load(outp)

    # only consider data within lane
    print('Trimming data to include only vehicles..', flush=True)

    def get_within_lane(idf):
        lane_bool = cs.within_this_type(
            type_name='lane',
            xlocs=idf['PositionX'].values,
            ylocs=idf['PositionY'].values
        )
        return idf[lane_bool]
    rdf1c = get_within_lane(rdf1)
    rdf2c = get_within_lane(rdf2)

    # # motion model
    # mm = CV2DRW2D()
    # mm.dt = 0.05
    # mm.phi = dict(sigma_vx=1.5, sigma_vy=1.5, sigma_w=0.2, sigma_l=0.2)
    # mm.update_matrices()
    # print(mm, flush=True)

    # # measurement model for radar
    # omr = LinearObservationModel(
    #     nx=mm.nx,
    #     observed_state_inds=[0, 2, 4, 5],
    #     name='Radar',
    #     ignore_inds_for_loglik=[2, 3]
    # )
    # omr.state_names = mm.state_names
    # omr.R = np.diag(np.array([1., 1., 0.2, 0.2])**2)
    # print(omr, flush=True)

    # motion model
    mm = CTRA_RECT()
    mm.dt = 0.05
    mm.phi = dict(
        sigma_omega=0.02,
        sigma_accn=0.1,
        sigma_w=0.2,
        sigma_l=0.2,
        min_speed=0.2
    )
    # mm.update_matrices()
    print(mm, flush=True)

    omr = LinearObservationModel(
        nx=mm.nx,
        observed_state_inds=[0, 1, 2, 3, 6, 7],
        name='CTRA_Linear'
    )
    omr.ignore_inds_for_loglik = [2, 5]
    omr.R = np.diag(np.array([1.5, 1.5, 15*np.pi/180., 0.5, 0.25, 0.25])**2)
    omr.state_names = mm.state_names
    print(omr)

    # kalman filter object
    # kf_base = KalmanFilter(
    #     nx=mm.nx,
    #     ny=omr.ny,
    #     dt=mm.dt,
    #     mat_F=mm.F,
    #     mat_Q=mm.Q,
    #     mat_H=omr.H,
    #     dt_tol=mm.dt/2 + 1e-4,
    #     state_names=mm.state_names
    # )
    # kf_base._P = np.diag(np.array([2., 2., 0.5, 0.5, 0.1, 0.1])**2)
    # print(kf_base, flush=True)

    kf_base = UnscentedKalmanFilter(
        nx=mm.nx,
        ny=omr.ny,
        dt=mm.dt,
        pars=dict(alpha=0.0001, beta=2, kappa=0, use_cholesky=False),
        fun_f=partial(mm.func_f, dt=mm.dt),
        fun_Q=partial(mm.func_Q, dt=mm.dt),
        mat_H=omr.H,
        dt_tol=mm.dt/2. + 0.0001,
        state_names=mm.state_names
    )
    kf_base.x_subtract = partial(subtract_func, angle_index=2)
    kf_base.y_subtract = partial(subtract_func, angle_index=2)
    kf_base.x_mean_fn = partial(mean_func, angle_index=2)
    kf_base.y_mean_fn = partial(mean_func, angle_index=2)
    kf_base._P = np.diag(
        np.array([2., 2., 20*np.pi/180, 0.5, 0.1, 0.1, 0.2, 0.2])**2)
    print(kf_base, flush=True)

    # set up fusion list
    olm = ObjectLifespanManager(
        loglik_threshold=-20,
        pred_lifespan=10.,
        min_lifespan=2.  # need a function here to respect the intersection
    )
    fusion_engine = MultisensorFusionEngine(
        kf_base=kf_base,
        lifespan_manager=olm,
        verbose=False
    )
    print(fusion_engine)

    # break the data into smaller chunks
    min_time = min(rdf1c.TimeElapsed.min(), rdf2c.TimeElapsed.min())
    max_time = max(rdf1c.TimeElapsed.max(), rdf2c.TimeElapsed.max())
    tlist = np.linspace(min_time, max_time, int((max_time-min_time)/60.))
    list_of_intervals = [(ix, iy) for ix, iy in zip(tlist[:-1], tlist[1:])]
    list_of_intervals = [(i, *ix) for i, ix in enumerate(list_of_intervals)]
    # print(list_of_intervals)
    # object list for fusion

    def radar_init_fn():
        fig, ax = initialize_traffic_plot(left_str='Econolite Radars')
        ax.set_xlim([-100, 50])
        ax.set_ylim([-50, 100])
        cs.plot_this_type(ax, 'lane', fc='gray', alpha=0.1, ec='none')
        ax.set_aspect('equal')
        return fig, ax

    radar_frame_fn = partial(
        create_frames,
        init_fn=radar_init_fn,
        ncores=1,
        verbose=False,
        list_of_fc=['r', 'b']
    )

    def fusion_init_fn():
        fig, ax = initialize_traffic_plot(left_str='IPC Sensor Fusion')
        ax.set_xlim([-100, 50])
        ax.set_ylim([-50, 100])
        cs.plot_this_type(ax, 'lane', fc='gray', alpha=0.1, ec='none')
        ax.set_aspect('equal')
        return fig, ax

    fusion_frame_fn = partial(
        create_frames,
        init_fn=fusion_init_fn,
        ncores=1,
        verbose=False,
        list_of_fc=['g'],
    )

    mp_fn = partial(
        run_sim,
        sensor_dfs=[rdf1c, rdf2c],
        obs_models={'Radar 1': omr, 'Radar 2': omr},
        frame_fns=[radar_frame_fn, fusion_frame_fn],
        output_dir=save_dir,
        time_res=0.5,
        fusion_engine=fusion_engine
    )

    results = run_loop(
        mp_fn, list_of_intervals[:12], desc=f'Fusion', ncores=36)
    # print(results[0])
    fused_df = pd.concat(results)
    fused_df.to_pickle(os.path.join(base_dir, f'fused_radars.pkl'))
