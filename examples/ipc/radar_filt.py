#!/usr/bin/env python
# coding: utf-8


from scipy.linalg import block_diag
from dataclasses import fields
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functools import partial
import datetime
import pickle
import matplotlib.animation as animation
# import cartopy.crs as ccrs
# from bayesfilt.telemetry.utils import *


from bayesfilt.ipc import *
from bayesfilt.models import *
from bayesfilt.filters import *


# In[4]:


# base_dir = '/lustre/eaglefs/projects/aumc/rsandhu/camera_radar_28aug2023'
base_dir = '/home/rsandhu/IpcFusion/camera_radar_fusion'
output_dir = os.path.join(base_dir, 'output')
fig_dir = os.path.join(base_dir, 'figs')
data_dir = os.path.join(base_dir, 'output', 'Transformed')
data_dir = os.path.join(base_dir, 'output', 'Aug_28_2023')


# In[5]:


# df_c1 = pd.read_pickle(os.path.join(data_dir, 'camera1_feb16.pkl'))
# df_c2 = pd.read_pickle(os.path.join(data_dir, 'camera2_feb16.pkl'))
df_r1 = pd.read_pickle(os.path.join(output_dir, 'radar_1_tr.pkl'))
df_r2 = pd.read_pickle(os.path.join(output_dir, 'radar_2_tr.pkl'))
# df_gps = pd.read_pickle(os.path.join(data_dir, 'gps.pkl'))


# In[6]:


df_r1.info()


# In[7]:


radar1 = TrafficSensor(
    name='Radar1',
    clr='m',
    tdata=df_r1.Time,
    # object_id=df_r1.ObjectID,
    xdata=df_r1.PositionX,
    ydata=df_r1.PositionY,
    heading_deg=-df_r1.Heading + 90.,
    speed=df_r1.Velocity,
    width=df_r1.Width,
    length=df_r1.Length
)
radar1.df.info()
radar2 = TrafficSensor(
    name='Radar2',
    clr='c',
    tdata=df_r2.Time,
    # object_id=df_r2.ObjectID,
    xdata=df_r2.PositionX,
    ydata=df_r2.PositionY,
    heading_deg=-df_r2.Heading + 90.,
    speed=df_r2.Velocity,
    width=df_r2.Width,
    length=df_r2.Length
)
# radar2.df.info()


# In[8]:


base_time = datetime.datetime(2023, 8, 28, 11, 23, 20)
sensors = [radar1, radar2]
for isensor in sensors:
    isensor.compute_timeelapsed(base_time=base_time, round_to='50ms')
    isensor.ignore_slow_moving_detections(cutoff_speed=4.)


# In[9]:


fig, ax = fig_init_fn_csprings()
for isensor in sensors:
    ax.plot(isensor.df.PositionX, isensor.df.PositionY, '.',
            color=isensor.clr,  markersize=0.1, alpha=0.99, label=radar1.name)
# tint.plot_this_zone(ax,iname='world')
ax.legend()
leg = ax.legend(handlelength=1.5, loc=3, borderaxespad=0)
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_markersize(5.)
ax.grid(True)
plt.show(fig)


# In[10]:


# motion model
dt = 0.05
mm = CA_RW(
    dof_ca=2,
    sigma_accn=[1., 1.],
    dof_rw=2,
    sigma_rw=[0.25, 0.25],
    xnames=['PositionX', 'SpeedX', 'AccnX', 'PositionY',
            'SpeedY', 'AccnY', 'Width', 'Length']
)
print(mm)


# In[11]:


mm.mat_Q(dt=0.1).shape


# In[12]:


om_radar = LinearObservationModel(nx=mm.nx, obs_state_inds=[
                                  0, 1, 3, 4, 6, 7], xnames=mm.xnames, name='linear')
# om_radar.ignore_inds_for_loglik = [2,3,4,5]
# om_radar.R = np.diag([ix**2 for ix in [0.5, 2., 0.5, 2., 0.25, 0.25]])
# om_radar.state_names = mm.state_names
print(om_radar)


# In[13]:


time_range = (260, 320)
olist_df1 = radar1.get_object_list(
    varnames=om_radar.ynames,
    timeelapsed_range=time_range
)
olist_df2 = radar2.get_object_list(
    varnames=om_radar.ynames,
    timeelapsed_range=time_range
)
olist_df = pd.concat([olist_df1, olist_df2])
olist_df.set_index('TimeElapsed', inplace=True)
olist_df.sort_index(inplace=True)
olist_df.reset_index(drop=False, inplace=True)
olist_df.info()


# In[14]:


kf_base = KalmanFilter(
    nx=mm.nx,
    ny=om_radar.ny,
    dt=dt,
    mat_F=mm.mat_F(dt),
    mat_Q=mm.mat_Q(dt),
    mat_H=om_radar.Hmat,
    start_P=np.diag(np.array([4., 2., 1., 4., 2., 1., 0.2, 0.2])**2),
    xnames=mm.xnames,
)
print(kf_base)


# In[15]:


kf_base.vars


# In[30]:


ls_manager = ObjectLifespanManager(
    loglik_threshold=-135,
    pred_lifespan=1.,
)
fusion_engine = MultisensorFusionEngine(
    kf_base=kf_base,
    lifespan_manager=ls_manager,
    verbose=False
)
fusion_engine.run(
    list_of_time=olist_df.TimeElapsed,
    list_of_detections=olist_df.Data,
    list_of_sensors=olist_df.Sensor,
    dict_of_start_mean_funcs={
        'Radar1': lambda y: om_radar.Hmat.T@y,
        'Radar2': lambda y: om_radar.Hmat.T@y
    },
    dict_of_obs_covariance={
        'Radar1': np.diag([0.5, 2., 0.5, 2., 0.25, 0.25])**2,
        'Radar2': np.diag([0.5, 2., 0.5, 2., 0.25, 0.25])**2
    },
)
fusion_engine.assemble_dataframe(
    xnames=mm.xnames, variance=True, metrics=True, remove_forecast_at_update=True)
fusion_engine


# In[31]:


fusion_engine.df.info()


# In[32]:


idf = fusion_engine.df
fused = TrafficSensor(
    name='Fused',
    clr='g',
    tdata=base_time +
        np.vectorize(timedelta)(seconds=idf.TimeElapsed.values.astype(float)),
    object_id=idf.ObjectId,
    xdata=idf.PositionX,
    ydata=idf.PositionY,
    speed=np.sqrt(idf.SpeedX**2 + idf.SpeedY**2),
    speed_x=idf.SpeedX,
    speed_y=idf.SpeedY,
    width=idf.Width,
    length=idf.Length,
    uncertainty=1. / np.sqrt(idf.PositionX_Var + idf.PositionY_Var)
)


# In[33]:


fused.df.info()


# In[34]:


sensors = [radar1, radar2, fused]
for isensor in sensors:
    isensor.compute_timeelapsed(base_time=datetime.datetime(
        2023, 8, 28, 11, 23, 20), round_to='50ms')
fig, ax = fig_init_fn_csprings()
for isensor in sensors:
    ax.plot(isensor.df.PositionX, isensor.df.PositionY, '.',
            color=isensor.clr,  markersize=0.1, alpha=0.99, label=isensor.name)
# tint.plot_this_zone(ax,iname='world')
ax.legend()
leg = ax.legend(handlelength=1.5, loc=3, borderaxespad=0)
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_markersize(5.)
ax.grid(True)
plt.show(fig)


# In[35]:


fig, ax = plt.subplots(2, 1, figsize=(12, 6))
for isensor in sensors:
    ax[0].plot(isensor.df.TimeElapsed, isensor.df.SpeedX, '.',
               markersize=0.92, label=isensor.name, color=isensor.clr)
    ax[1].plot(isensor.df.TimeElapsed, isensor.df.PositionY, '.',
               markersize=0.92, label=isensor.name, color=isensor.clr)
ax[0].set_ylim([-100, 100])
ax[1].set_ylim([-20, 20])
for iax in ax:
    iax.set_xlim(*time_range)
    # ax.set_xlim([330,380])
    # ax.set_ylim([0,5])
    iax.grid(True)
    iax.legend()
    iax.set_xlabel('Time elapsed [sec]')
# ax.set_ylabel('X Position [m]')
plt.show()


# In[36]:


min_etime, max_etime = time_range
# min_etime, max_etime = (800,1000)
time_res = 0.2
nframes = int((max_etime-min_etime)/time_res) + 1
list_of_times = np.linspace(min_etime, max_etime, nframes)
# list_of_times
image_dir = os.path.join(base_dir, 'output', 'run1')
create_frames_from_sensors(
    list_of_sensors=sensors,
    list_of_etimes=list_of_times,
    out_dir=image_dir,
    fig_init_fn=partial(fig_init_fn_csprings, True),
    ncores=1
)
create_video_from_frames(
    image_dir=image_dir,
    fps=int(1/time_res),
    delete_images_after=True
)


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


idf = fusion_engine.df
idf['TimeElapsed'] = idf.TimeElapsed.round(2)
idf = idf.loc[idf.MetricYresNorm.notna(), :]
idf.set_index(['ObjectId', 'TimeElapsed'], inplace=True, drop=True)
idf.sort_index(inplace=True)
idf.head()
idf.info()


# In[ ]:


jdf = fusion_engine.dfa
jdf = jdf.loc[jdf.DaLogLik.notna(), :]
jdf.set_index(['ObjectId', 'TimeElapsed'], inplace=True, drop=True)
jdf.sort_index(inplace=True)
jdf.info()


# In[ ]:


result = pd.merge(idf, jdf, on=['ObjectId', 'TimeElapsed'])
result.info()


# In[ ]:


pd.merge(idf, jdf, how='left', left_index=True, right_index=True)


# In[ ]:


kdf = pd.concat([idf, jdf], axis=1)
kdf.info()


# In[ ]:


xx = FilterMetrics()


# In[ ]:


kdf = pd.merge(idf, jdf, how='outer', on=('ObjectId', 'TimeElapsed'))


# In[ ]:


kdf.info()


# In[ ]:


kdf.head()


# In[ ]:


idf.index[:30]


# In[ ]:


jdf = fusion_engine.df_da
jdf = jdf.loc[jdf.DaLogLik.notna(), :]
jdf.set_index(['ObjectId', 'TimeElapsed'], inplace=True, drop=True)
jdf.sort_index(inplace=True)
jdf.head()


# In[ ]:


jdf.info()


# In[ ]:


kdf = idf.merge(jdf, left_index=True, right_index=True)


# In[ ]:


kdf.info()


# In[ ]:


# merge the two dataframe and then do what you want


# In[ ]:


kdf = pd.merge(
    left=idf,
    right=jdf,
    left_index=True,
    right_index=True
)
kdf.shape


# In[ ]:


idf = pd.DataFrame(fusion_engine._history_da)
idf.head()
fusion_engine._history_kf[1234] = None
fusion_engine._history_kf


# In[ ]:


sensors = [radar1, radar2, fused]
for isensor in sensors:
    isensor.compute_timeelapsed(base_time=datetime.datetime(
        2023, 8, 28, 11, 23, 20), round_to='50ms')
fig, ax = fig_init_fn_csprings()
for isensor in sensors:
    ax.plot(isensor.df.PositionX, isensor.df.PositionY, '.',
            color=isensor.clr,  markersize=0.1, alpha=0.99, label=isensor.name)
# tint.plot_this_zone(ax,iname='world')
ax.legend()
leg = ax.legend(handlelength=1.5, loc=3, borderaxespad=0)
for lh in leg.legend_handles:
    lh.set_alpha(1)
    lh.set_markersize(5.)
ax.grid(True)
plt.show(fig)


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(12, 6))
for isensor in sensors:
    ax[0].plot(isensor.df.TimeElapsed, isensor.df.PositionX, '.',
               markersize=0.92, label=isensor.name, color=isensor.clr)
    ax[1].plot(isensor.df.TimeElapsed, isensor.df.SpeedX, '.',
               markersize=0.92, label=isensor.name, color=isensor.clr)
ax[0].set_ylim([-100, 100])
ax[1].set_ylim([-20, 20])
for iax in ax:
    iax.set_xlim(*time_range)
    # ax.set_xlim([330,380])
    # ax.set_ylim([0,5])
    iax.grid(True)
    iax.legend()
    iax.set_xlabel('Time elapsed [sec]')
# ax.set_ylabel('X Position [m]')
plt.show()


# In[ ]:


min_etime, max_etime = time_range
# min_etime, max_etime = (800,1000)
time_res = 0.2
nframes = int((max_etime-min_etime)/time_res) + 1
list_of_times = np.linspace(min_etime, max_etime, nframes)
# list_of_times
image_dir = os.path.join(base_dir, 'output', 'run1')
create_frames_from_sensors(
    list_of_sensors=sensors,
    list_of_etimes=list_of_times,
    out_dir=image_dir,
    fig_init_fn=partial(fig_init_fn_csprings, True),
    ncores=1,
    alpha=0.75
)
create_video_from_frames(
    image_dir=image_dir,
    fps=int(1/time_res),
    delete_images_after=True
)


# In[ ]:


# In[ ]:


idf = radar1.df.copy()
idf.set_index(['ObjectId', 'TimeElapsed'], inplace=True)
idf.sort_index(inplace=True)
idf.iloc[750:]


# In[ ]:


np.zeros(5).shape


# In[ ]:


# when object is dead get rid of predicted


# In[ ]:


xx = FilterMetrics()
xx


# In[ ]:


xx.XresNorm = 100


# In[ ]:


xx


# In[ ]:


xx.as_dict()


# In[ ]:


for v in fields(xx):
    print(v.name, getattr(xx, v.name))


# In[ ]:


xx._reset_to_default()


# In[ ]:


xx


# In[ ]:


list(np.hstack([[f'X{i}', f'P{i}'] for i in range(3)]))


# In[ ]:


_mat = np.eye(2)
_mat[0, 1] = dt
_mat


# In[ ]:


np.block_diag([_mat for _ in range(3)])


# In[ ]:


np.kron(np.diag([1, 2, 3]), np.ones((2, 2)))


# In[ ]:


xx = np.random.randn(9).reshape(3, 3)
yy = np.random.randn(16).reshape(4, 4)


# In[ ]:


block_diag(xx, yy).shape


# In[ ]:
