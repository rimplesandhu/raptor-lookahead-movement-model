"""Usefull functions for multisensor fusion engine"""
# pylint: disable=invalid-name
import glob
from typing import Optional, Union, Dict, List
from pathlib import Path
import os
from functools import partial
import datetime
from numpy import ndarray
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
from bayesfilt.telemetry.utils import run_loop

cb_clrs = ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']


# def get_short_df(df_dict, start_time = time(10,46), end_time = time(10,47)):
#     dfshort = {}
#     for key, idf in df_dict.items():
#         dfshort[key] = idf[idf['TimeLocal'].dt.time.between(start_time, end_time)]
#     return dfshort

# def transform_clockwise(Xval, Yval, angle=0.):
#     """Rotate and translate a cartesian coordinate system """
#     angle_rad = np.radians(angle)
#     Xval_new = Xval * np.cos(angle_rad) - Yval * np.sin(angle_rad)
#     Yval_new = Xval * np.sin(angle_rad) + Yval * np.cos(angle_rad)
#     return Xval_new, Yval_new

# def translate2D(Xval, Yval, Xmove:float, Ymove:float):
#     """Rotate and translate a cartesian coordinate system """
#     angle_rad = np.radians(angle)
#     Xval_new = Xval * np.cos(angle_rad) - Yval * np.sin(angle_rad)
#     Yval_new = Xval * np.sin(angle_rad) + Yval * np.cos(angle_rad)
#     return Xval_new, Yval_new


def change_angle(in_angle):
    """Transform the angle for creating animations"""
    out_angle = -in_angle + 90
    out_angle[out_angle > 180] -= 360
    return out_angle


def get_frame_fname(ix):
    """Returns filename for a given frame id"""
    return f'frame.{str(ix).zfill(6)}.png'


def create_video_from_images(
    image_dir: str,
    fps: int,
    filename: str = 'output.avi'
):
    """Creates/saves video from images"""
    # print(image_dir, filename)
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(image_dir, images[0]))
    height, width, layers = frame.shape
    fncc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(os.path.join(
        image_dir, filename), fncc, fps, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))
        os.remove(os.path.join(image_dir, image))
    cv2.destroyAllWindows()
    video.release()


def initialize_traffic_plot(
    right_str: str = None,
    left_str: str = None,
    clr: str = 'lavender'
):
    """Initialize the plot for creating animations"""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    ax.axis(False)
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor(clr)
    fig.tight_layout()

    if right_str is not None:
        ax.text(1., 1., str(right_str), ha='right', va='top',
                color='k', transform=ax.transAxes,
                bbox=dict(fc=clr, ec=clr))
    if left_str is not None:
        ax.text(0., 1., str(left_str), ha='left', va='top',
                color='k', transform=ax.transAxes,
                bbox=dict(fc=clr, ec=clr))
    return fig, ax


def merge_frames(dir1, dir2, out_dir):
    """Merge frames"""
    flist1 = glob.glob(os.path.join(dir1, "*.png"))
    flist2 = glob.glob(os.path.join(dir2, "*.png"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for file1, file2 in zip(flist1, flist2):
        image1 = cv2.imread(file1)
        image2 = cv2.imread(file2)
        vis = np.concatenate((image1, image2), axis=1)
        fout = os.path.join(out_dir, file1.split('/')[-1])
        cv2.imwrite(fout, vis)


def create_frames(
    list_of_dfs: list[pd.DataFrame],
    list_of_etimes: list[float],
    image_dir: str,
    init_fn: callable,
    verbose=False,
    ncores=36,
    list_of_fc=[],
):
    """Create animation from a list of dataframes"""
    istr = f'{round(list_of_etimes[0],3)}-{round(list_of_etimes[-1],3)}'
    if verbose:
        print(f'Creating frames {istr}..')
    filelist = glob.glob(os.path.join(image_dir, "*.png"))
    _ = [os.remove(f) for f in filelist]
    time_res = list_of_etimes[1] - list_of_etimes[0]
    list_of_tuple = [(ix, iy, None, None)
                     for ix, iy in enumerate(list_of_etimes)]
    for i, idf in enumerate(list_of_dfs):
        frame_fn = partial(
            draw_frame,
            tlocs=idf['TimeElapsed'].values,
            xlocs=idf['PositionX'].values,
            ylocs=idf['PositionY'].values,
            headings=idf['Heading'].values,
            ids=idf['ObjectID'].values,
            time_pad=time_res/2.,
            case_dir=image_dir,
            initialize_fn=init_fn,
            widths=idf['Width'],
            lengths=idf['Length'],
            ec='none',
            fc=list_of_fc[i],
        )
        if verbose:
            results = run_loop(
                frame_fn,
                list_of_tuple,
                desc=f'df{i}',
                ncores=ncores,
                disable=False
            )
        else:
            results = run_loop(
                frame_fn,
                list_of_tuple,
                desc=f'df{i}',
                ncores=ncores,
                disable=True
            )

        list_of_tuple = results.copy()


def draw_frame(
    frame_tuple: tuple[int, float],
    tlocs: ndarray,
    xlocs: ndarray,
    ylocs: ndarray,
    headings: ndarray,
    case_dir: str,
    time_pad: float,
    initialize_fn: callable,
    widths: ndarray | None = None,
    lengths: ndarray | None = None,
    alphas: ndarray | None = None,
    ids: ndarray | None = None,
    **kwargs
):
    # print(f'{ix}-', end="", flush=True)
    Path(case_dir).mkdir(parents=True, exist_ok=True)
    frame_id, at_time, fig, ax = frame_tuple
    ibool = (tlocs > (at_time - time_pad)) & (tlocs < (at_time + time_pad))
    if fig is None:
        fig, ax = initialize_fn()
    widths = np.ones_like(xlocs)*2.5 if widths is None else widths
    lengths = np.ones_like(xlocs)*5 if lengths is None else lengths
    ids = np.ones_like(xlocs, dtype=int)*0 if ids is None else ids
    alphas = np.ones_like(xlocs)*0.75 if alphas is None else alphas
    jdf = pd.DataFrame({
        'ObjectId': ids[ibool],
        'PositionX': xlocs[ibool],
        'PositionY': ylocs[ibool],
        'Heading': headings[ibool],
        'Width': widths[ibool],
        'Length': lengths[ibool],
        'Alpha': alphas[ibool]
    })
    jdf.drop_duplicates(['ObjectId'], keep='last', inplace=True)
    for _, irow in jdf.iterrows():
        centroid = (irow['PositionX'], irow['PositionY'])
        rot_transform = ax.transData
        lowerleft_loc = (
            centroid[0] - irow['Width'] / 2.,
            centroid[1] - irow['Length'] / 2.
        )
        # angle = (irow['Heading'] + 90.) % (360.)
        angle = -irow['Heading']
        rot_transform = mpl.transforms.Affine2D().rotate_deg_around(
            centroid[0],
            centroid[1],
            angle
        ) + ax.transData
        ibox = patches.FancyBboxPatch(
            xy=lowerleft_loc,
            width=irow['Width'],
            height=irow['Length'],
            boxstyle=patches.BoxStyle('round', rounding_size=1.),
            transform=rot_transform,
            alpha=irow['Alpha'],
            **kwargs
        )
        ax.add_patch(ibox)
        ax.text(
            irow['PositionX'],
            irow['PositionY'],
            str(int(irow['ObjectId']))[-2:],
            fontsize=4,
            ha='center',
            va='center',
            color='k'
        )
    Path(case_dir).mkdir(parents=True, exist_ok=True)
    fname = get_frame_fname(frame_id)
    fig.savefig(os.path.join(case_dir, fname),
                dpi=160, bbox_inches='tight')
    plt.close()
    return (frame_id, at_time, fig, ax)


def add_colored_legend(ax, idict):
    """add legend"""
    for ix, (iname, iclr) in enumerate(idict.items()):
        ax.text(.0, 0.01 + ix * 0.05, iname,
                fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc=iclr, ec='none', alpha=0.25))


def get_list_of_times(min_time, max_time, time_res_ms, this_date: str = None):
    """Returns list of times given min, max time and time resolution"""
    this_date = datetime.datetime.today().date() if this_date is None else this_date
    start_time = datetime.datetime.combine(this_date, min_time)
    end_time = datetime.datetime.combine(this_date, max_time)
    time_duration = (end_time - start_time).total_seconds()
    num_of_frames = int(time_duration * 1000 // time_res_ms) + 1
    list_of_tdeltas = [datetime.timedelta(
        milliseconds=i * time_res_ms) for i in range(num_of_frames)]
    list_of_times = [start_time + idelta for idelta in list_of_tdeltas]
    return list_of_times


# def box_function(irow, rot_start, box_fill: bool = True, box_color='r'):
#     iangle = (irow['Heading'] + np.pi / 2) % (2.0 * np.pi)
#     coords = (irow['X'] - irow['Width'] / 2, irow['Y'] - irow['Length'] / 2.)
#     coords_rotate = rot_start.transform((irow['X'], irow['Y']))
#     rot_this = mpl.transforms.Affine2D().rotate_deg_around(
#         coords_rotate[0],
#         coords_rotate[1],
#         np.degrees(iangle)
#     )

#     ibox = patches.FancyBboxPatch(coords,
#                                   irow['Width'], irow['Length'],
#                                   fill=box_fill,
#                                   ec=box_color, fc=box_color,
#                                   alpha=0.8, lw=1.25,
#                                   boxstyle='round',
#                                   linewidth=0.9, transform=rot_start + rot_this)
#     return ibox


# def save_frame(
#     etime,
#     sensor_df_dict,
#     fusion_df,
#     cs,
#     buffer_time=0.03
# ):
#     fig, ax = plt.subplots(figsize=(6, 6))
#     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
#     cs.plot_this_zone(ax, 'world', alpha=0.01, fc='g', ec='none')
#     cs.plot_this_type(ax, 'radar', '*k', markersize=8.)
#     # cs.plot_this_type(ax, 'radar_coverage', fc = 'r', ec='none', alpha=0.05)
#     cs.plot_this_zone(ax, 'radar_silver_coverage',
#                       fc='b', ec='none', alpha=0.05)
#     cs.plot_this_zone(ax, 'radar_black_coverage',
#                       fc='r', ec='none', alpha=0.05)
#     ax.set_xlim([cs.extent[0], cs.extent[1]])
#     ax.set_ylim([cs.extent[2], cs.extent[3]])
#     ax.set_aspect('equal')
#     rdf1_bool = sensor_df_dict['r1']['TimeElapsed'].between(
#         etime - buffer_time, etime + buffer_time)
#     rdf1_short = sensor_df_dict['r1'][rdf1_bool]
#     rdf2_bool = sensor_df_dict['r2']['TimeElapsed'].between(
#         etime - buffer_time, etime + buffer_time)
#     rdf2_short = sensor_df_dict['r2'][rdf2_bool]
#     fdf_short = fusion_df[fusion_df['TimeElapsed'].between(
#         etime - buffer_time, etime + buffer_time)]
#     # print(rdf1_short.shape[0], rdf2_short.shape[0], fdf_short.shape[0])
#     for _, irow in rdf1_short.iterrows():
#         ax.add_patch(box_function(irow, box_fill=False,
#                      box_color='r', rot_start=ax.transData))
#     for _, irow in rdf2_short.iterrows():
#         ax.add_patch(box_function(irow, box_fill=False,
#                      box_color='b', rot_start=ax.transData))
#     for _, irow in fdf_short.iterrows():
#         ax.add_patch(box_function(irow, box_fill=True,
#                      box_color='g', rot_start=ax.transData))
#         ax.text(irow['X'], irow['Y'], str(irow['ObjectId']),
#                 horizontalalignment='center', verticalalignment='center',
#                 color='w', fontsize=10)
#     return fig, ax


# def create_frame_pngs(
#     case_dir,
#     dt: float,
#     sensor_df_dict,
#     fusion_df,
#     cs
# ):
#     time_range = (fusion_df['TimeElapsed'].min(),
#                   fusion_df['TimeElapsed'].max())
#     time_list = np.round(np.linspace(time_range[0], time_range[1], int(
#         (time_range[1] - time_range[0]) / dt) + 1), 2)
#     for i, itime in enumerate(time_list):
#         print(f'{i}-', end="")
#         fig, ax = save_frame(itime, sensor_df_dict=sensor_df_dict,
#                              fusion_df=fusion_df, cs=cs)
#         fname = f'frame.{str(i).zfill(5)}.png'
#         fig.savefig(os.path.join(case_dir, fname), dpi=100)
#         plt.close()


# def get_corners_from_center_and_width(
#         center: Tuple[float, float],
#         width: Tuple[float, float]
# ) -> Sequence[Tuple[float, float]]:
#     """Get corners of rectangule from center and width"""
#     left_bottom = (center[0] - width[0] / 2, center[1] - width[1] / 2)
#     right_bottom = (center[0] + width[0] / 2, center[1] - width[1] / 2)
#     right_upper = (center[0] + width[0] / 2, center[1] + width[1] / 2)
#     left_upper = (center[0] - width[0] / 2, center[1] + width[1] / 2)
#     return [left_bottom, right_bottom, right_upper, left_upper]

#     # @df.setter
#     # def df(self, val) -> None:
#     #     asser
#     #     if val.shape != self._df.shape:
#     #         print('KalmanFilter: Shape mismatch while setting P matrix!')
#     #         raise ValueError(f'Desired: {self._P.shape} Input: {val.shape}')
#     #     self._P = val


# def transform_cartesian(point, angle=0., origin=(0, 0)):
#     """Rotate and translate a cartesian coordinate system """
#     angle_rad = np.radians(angle % 360)
#     # Shift the point so that origin becomes the origin
#     new_point = (point[0] - 0 * origin[0], point[1] - 0 * origin[1])
#     new_point = (new_point[0] * np.cos(angle_rad) - new_point[1] * np.sin(angle_rad),
#                  new_point[0] * np.sin(angle_rad) + new_point[1] * np.cos(angle_rad))
#     # Reverse the shifting we have done
#     new_point = (new_point[0] + origin[0],
#                  new_point[1] + origin[1])
#     return new_point


# def plot_csprings_parking_lot(ax):
#     cs = get_csprings_parking_lot()
#     cs.plot_this_zone(ax, 'world', alpha=0.01, fc='g', ec='none')
#     cs.plot_this_type(ax, 'radar', '*k', markersize=8.)
#     # cs.plot_this_type(ax, 'radar_coverage', fc = 'r', ec='none', alpha=0.05)
#     cs.plot_this_zone(ax, 'radar_silver_coverage',
#                       fc='b', ec='none', alpha=0.05)
#     cs.plot_this_zone(ax, 'radar_black_coverage',
#                       fc='r', ec='none', alpha=0.05)


# def animate_this_source(
#     ax,
#     etime: ndarray,
#     xloc: ndarray,
#     yloc: ndarray,
#     yaw: Optional[ndarray] = None,
#     width: Union[ndarray, float] = 1.,
#     length: Union[ndarray, float] = 1.,
#     object_ids: Optional[ndarray] = None,
#     obj_color: str = 'r',
#     box_fill: bool = True,
#     alphas: Optional[ndarray] = None,
#     num_of_objects: int = 50,
#     t_buffer: float = 0.1
# ):
#     """ Animates the data source """
#     assert xloc.size == yloc.size, 'Size mismatch b/w xloc and yloc!'
#     width = np.full_like(xloc, width) if isinstance(width, float) else width
#     length = np.full_like(xloc, length) if isinstance(
#         length, float) else length

#     # box overlay
#     def get_box_object():
#         return patches.FancyBboxPatch((0, 0), 0., 0., fill=box_fill,
#                                       ec=obj_color, fc=obj_color,
#                                       alpha=0.8, lw=1.25,
#                                       boxstyle='round',
#                                       linewidth=0.9)
#     boxes = []
#     if yaw is not None:
#         boxes = [get_box_object() for _ in range(num_of_objects)]
#         for ibox in boxes:
#             ibox.set_boxstyle('round', rounding_size=1.)
#             ax.add_patch(ibox)

#     # text overlay
#     def get_text_object():
#         return ax.text(0., 0., '', horizontalalignment='center',
#                        verticalalignment='center',
#                        color='w', fontsize=8)
#     texts = []
#     if object_ids is not None:
#         texts = [get_text_object() for _ in range(num_of_objects)]

#     msize = 5. if yaw is None else 1.
#     points, = ax.plot([], 'o' + obj_color, alpha=0.5, markersize=msize)
#     rot_start = ax.transData

#     def animate_func(itime: float):
#         tbool = np.logical_and(
#             etime >= itime - t_buffer,
#             etime <= itime + t_buffer
#         )
#         inds = np.where(tbool)[0]
#         points.set_data(xloc[inds], yloc[inds])

#         for i, indx in enumerate(inds):
#             iangle = (yaw[indx] + np.pi / 2) % (2.0 * np.pi)
#             coords = rot_start.transform((xloc[indx], yloc[indx]))
#             if object_ids is not None:
#                 obj_id = object_ids[indx]
#                 texts[i].set_text(str(obj_id))
#                 texts[i].set_x(xloc[indx])
#                 texts[i].set_y(yloc[indx])
#             if yaw is not None:
#                 boxes[i].set_x(xloc[indx] - width[indx] / 2)
#                 boxes[i].set_y(yloc[indx] - length[indx] / 2)
#                 boxes[i].set_width(width[indx])
#                 boxes[i].set_height(length[indx])
#                 rot_this = mpl.transforms.Affine2D().rotate_deg_around(
#                     coords[0],
#                     coords[1],
#                     np.degrees(iangle)
#                 )
#                 boxes[i].set_transform(rot_start + rot_this)
#                 if alphas is not None:
#                     boxes[i].set_alpha(alphas[indx])
#                 else:
#                     boxes[i].set_alpha(1.)
#         for jj in range(len(inds), num_of_objects):
#             if yaw is not None:
#                 boxes[jj].set_x(1000.)
#                 boxes[jj].set_y(1000.)
#                 # boxes[j].set_alpha(0.)
#                 boxes[jj].set_width(0.)
#                 boxes[jj].set_height(0.)
#             if object_ids is not None:
#                 texts[jj].set_text('')
#             # texts[j].set_x(1000)
#             # texts[j].set_y(1000)
#         return points, boxes,
#     return animate_func


# def merge_animation_functions(list_of_funcs):
#     """Merge multiple animate functions into one"""
#     def animate(itime):
#         points_list = []
#         boxes_list = []
#         for ifunc in list_of_funcs:
#             points, boxes = ifunc(itime)
#             points_list.append(points)
#             boxes_list.append(boxes)
#         return points_list, boxes_list,
#     return animate


# def animate_scene(sensor_df, fusion_df, time_list, extent):
#     """Animate radar data"""
#     fig, ax = plt.subplots(figsize=(6, 6))
#     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
#     plot_csprings_parking_lot(ax)
#     ax.set_xlim([extent[0], extent[1]])
#     ax.set_ylim([extent[2], extent[3]])
#     ax.set_aspect('equal')
#     list_of_funcs = []
#     list_of_funcs.append(animate_this_source(
#         ax,
#         etime=sensor_df['r1']['TimeElapsed'].values,
#         xloc=sensor_df['r1']['X'].values,
#         yloc=sensor_df['r1']['Y'].values,
#         yaw=sensor_df['r1']['Heading'].values,
#         width=sensor_df['r1']['Width'].values,
#         length=sensor_df['r1']['Length'].values,
#         object_ids=None,
#         obj_color='r',
#         box_fill=False,
#         alphas=None,
#         t_buffer=0.03
#     ))
#     list_of_funcs.append(animate_this_source(
#         ax,
#         etime=sensor_df['r2']['TimeElapsed'].values,
#         xloc=sensor_df['r2']['X'].values,
#         yloc=sensor_df['r2']['Y'].values,
#         yaw=sensor_df['r2']['Heading'].values,
#         width=sensor_df['r2']['Width'].values,
#         length=sensor_df['r2']['Length'].values,
#         object_ids=None,
#         obj_color='b',
#         box_fill=False,
#         alphas=None,
#         t_buffer=0.03
#     ))
#     list_of_funcs.append(animate_this_source(
#         ax,
#         etime=fusion_df['TimeElapsed'].values,
#         xloc=fusion_df['X'].values,
#         yloc=fusion_df['Y'].values,
#         yaw=fusion_df['Heading'].values,
#         width=fusion_df['Width'].values,
#         length=fusion_df['Length'].values,
#         object_ids=fusion_df['ObjectId'].values,
#         obj_color='g',
#         box_fill=True,
#         alphas=fusion_df['ErrorStrength'].values,
#         t_buffer=0.03
#     ))
#     animate = merge_animation_functions(list_of_funcs)
#     ani = animation.FuncAnimation(fig, animate, list(time_list),
#                                   interval=10, repeat=False)
#     # lgd_patches = [
#     #     patches.Patch(color='r', alpha=0.2, ec=None, label='Radar1'),
#     #     patches.Patch(color='b', alpha=0.2, ec=None, label='Radar2'),
#     #     # patches.Patch(color='k', alpha= 0.2, ec=None, label='Fusion Engine')
#     # ]
#     # ax.legend(handles=lgd_patches, borderaxespad=0.1, loc='upper right')
#   # ax.set_facecolor('floralwhite')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ttl = ax.text(.015, 0.97, 'Radar 1', fontsize=12, color='black',
#                   transform=ax.transAxes, va='center', ha='left',
#                   bbox=dict(boxstyle="round", facecolor='Red',
#                             ec='Red', alpha=0.25))
#     ttl = ax.text(.155, 0.97, 'Radar 2', fontsize=12, color='black',
#                   transform=ax.transAxes, va='center', ha='left',
#                   bbox=dict(boxstyle="round", facecolor='Blue',
#                             ec='Blue', alpha=0.25))
#     ttl = ax.text(.3, 0.97, 'IPC Fusion Engine', fontsize=12, color='black',
#                   transform=ax.transAxes, va='center', ha='left',
#                   bbox=dict(boxstyle="round", facecolor='Green',
#                             ec='Green', alpha=0.25))
#     return ani


# def animate_fusion(fusion_df, time_list, extent):
#     """Animate radar data"""
#     fig, ax = plt.subplots(figsize=(6, 6))
#     fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
#     ax.set_xlim([extent[0], extent[1]])
#     ax.set_ylim([extent[2], extent[3]])
#     ax.set_aspect('equal')
#     list_of_funcs = []
#     list_of_funcs.append(animate_this_source(
#         ax,
#         etime=fusion_df['TimeElapsed'].values,
#         xloc=fusion_df['X'].values,
#         yloc=fusion_df['Y'].values,
#         yaw=fusion_df['Heading'].values,
#         width=fusion_df['Width'].values,
#         length=fusion_df['Length'].values,
#         object_ids=fusion_df['ObjectId'].values,
#         obj_color='g',
#         box_fill=True,
#         alphas=fusion_df['ErrorStrength'].values,
#         t_buffer=0.05
#     ))
#     animate = merge_animation_functions(list_of_funcs)
#     ani = animation.FuncAnimation(fig, animate, list(time_list),
#                                   interval=10, repeat=False)
#     ax.set_facecolor('floralwhite')
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ttl = ax.text(.015, 0.97, 'IPC Fusion Engine', fontsize=12, color='white',
#                   transform=ax.transAxes, va='center', ha='left',
#                   bbox=dict(boxstyle="round", facecolor='olive'))
#     return ani


# import imageio


# def merge_two_gifs(fpath1, fpath2, fpath_out):
#     # Create reader object for the gif
#     gif1 = imageio.get_reader(fpath1)
#     gif2 = imageio.get_reader(fpath2)

#     # If they don't have the same number of frame take the shorter
#     number_of_frames = min(gif1.get_length(), gif2.get_length())

#     # Create writer object
#     new_gif = imageio.get_writer(fpath_out)

#     for frame_number in range(number_of_frames):
#         img1 = gif1.get_next_data()
#         img2 = gif2.get_next_data()
#         # here is the magic
#         new_image = np.hstack((img1, img2))
#         new_gif.append_data(new_image)

#     gif1.close()
#     gif2.close()
#     new_gif.close()

# # def animate_this_dataframe(ics, idf, iax, clr, box_fill=True):
#     """Animate this dataframe"""
#     eltime = idf['TimeElapsed'].values
#     xloc = idf['X'].copy().values
#     yloc = idf['Y'.copy().values
#     if 'X_var' in idf.columns:
#         xloc_var = idf['X_var'].copy().values
#         yloc_var = idf['Y_var'].copy().values
#         alphas = 1. / np.sqrt(xloc_var + yloc_var)
#         alphas /= np.amax(alphas)
#         alphas = np.clip(alphas, 0.2, 1.0)
#     else:
#         alphas = None
#     length = idf['Length'].copy().values
#     width = idf['Width'].copy().values
#     #col_name = ['zone', 'orientation']
#     object_ids = None
#     if 'ObjectId' in idf.columns:
#         object_ids = idf['ObjectId'].copy().tolist()
#     col_name = ['zone', 'orientation']
#     zone_pair = ics.df.loc[ics.df['orientation'].notna(), col_name].values
#     if 'yaw' in idf.columns:
#         angle = idf['yaw'].values
#     else:
#         angle_raw = np.arctan2(idf['x_vel'], idf['y_vel']) * 180. / np.pi
#         angle = angle_raw.copy().values
#         bool_speed = np.sqrt(idf['x_vel']**2 + idf['y_vel']**2) < 1.
#         angle[np.where(bool_speed)[0]] = np.nan
#         for izone, iangle in zone_pair:
#             ibool = [izone.contains(Point(ix, iy))
#                      for ix, iy in zip(xloc, yloc)]
#             angle[np.where(bool_speed & ibool)[0]] = iangle
#     return animate_this_source(iax, eltime, xloc, yloc, -angle,
#                                width, length, object_ids, clr, box_fill, alphas)


# def animate_this_dataframe(ics, idf, iax, clr, box_fill=True):
#     """Animate this dataframe"""
#     eltime = idf['TimeElapsed'].values
#     xloc = idf['X'].copy().values
#     yloc = idf['Y'.copy().values
#     if 'X_var' in idf.columns:
#         xloc_var = idf['X_var'].copy().values
#         yloc_var = idf['Y_var'].copy().values
#         alphas = 1. / np.sqrt(xloc_var + yloc_var)
#         alphas /= np.amax(alphas)
#         alphas = np.clip(alphas, 0.2, 1.0)
#     else:
#         alphas = None
#     length = idf['Length'].copy().values
#     width = idf['Width'].copy().values
#     #col_name = ['zone', 'orientation']
#     object_ids = None
#     if 'ObjectId' in idf.columns:
#         object_ids = idf['ObjectId'].copy().tolist()
#     heading = None
#     if 'Heading' in idf.columns:
#         angle = idf['yaw'].copy().values
#     return animate_this_source(iax, eltime, xloc, yloc, -angle,
#                                width, length, object_ids, clr, box_fill, alphas)


# def ipc_fusion_engine(
#     sensor_data_df: pd.DataFrame,
#     mm,
#     sensor_obs_models: Dict,
#     start_state_stds: List[float],
#     loglik_threshold: float,
#     lifespan_after_last_update: float = 10.,
#     min_lifespan: float = 1.,
#     verbose: bool = False
# ):
#     """ IPC fusion engine"""
#     verbose = False
#     object_list = {}
#     object_list.clear()
#     for _, row in sensor_data_df.iterrows():
#         this_sensor = row['Sensor']
#         this_time = np.around(row['TimeElapsed'], 3)
#         this_obs = np.asarray(row['ObjectData'])
#         if verbose:
#             print(f'--Incoming at {this_time} from sensor {this_sensor}')
#         om = sensor_obs_models[this_sensor]
#         loglik = {}
#         dead_object_ids = []
#         loglik.clear()
#         for this_id, this_kf in object_list.items():
#             if this_time - this_kf.last_update_at < lifespan_after_last_update:
#                 # print(f'dd{this_id} at {this_time},{this_kf.last_update_at}')
#                 this_kf.forecast_upto(this_time)
#                 this_kf.H = om.H.copy()
#                 this_kf.R = om.R.copy()
#                 loglik[this_id] = this_kf.get_loglik_of_obs(
#                     y_obs=this_obs.copy(),
#                     ignore_obs_inds=[4, 5]
#                 )
#             else:
#                 this_kf.forecast_upto(
#                     this_kf.last_update_at + lifespan_after_last_update)
#                 ltime = this_kf.last_update_at - this_kf.get_time_elapsed()[0]
#                 if ltime < min_lifespan:
#                     dead_object_ids.append(this_id)
#                 # print(f'{this_id} at {this_time},{this_kf.last_update_at}')
#         for this_id in dead_object_ids:
#             del object_list[this_id]
#         if verbose:
#             print('  Logliks: ', [np.around(v, 1) for k, v in loglik.items()])
#             print('  Updates: ', [np.around(v.last_update_at, 2)
#                   for k, v in object_list.items()])
#         probable_objects = dict((k, v)
#                                 for k, v in loglik.items() if v >= loglik_threshold)
#         if bool(probable_objects):
#             found_this_object = max(loglik, key=loglik.get)
#             object_list[found_this_object].obs = this_obs
#             object_list[found_this_object].update()
#         else:
#             found_this_object = len(object_list)
#             if verbose:
#                 print(f'  Creating {found_this_object} at {this_time}')
#             object_list[found_this_object] = KalmanFilter(
#                 mm.nx,
#                 ny=om.ny,
#                 dt=mm.dt,
#                 object_id=found_this_object
#             )
#             update_kf_matrices(object_list[found_this_object], mm, om)
#             start_state_mean = om.H.T @ this_obs
#             start_state_mean[7] = 5.
#             object_list[found_this_object].initiate_state(
#                 this_time,
#                 start_state_mean,
#                 np.diag(start_state_stds**2)
#             )
#     list_of_dfs = [
#         object_list[obj_id].df_filter for obj_id in list(object_list.keys())]
#     fusion_df = pd.concat(list_of_dfs)
#     fusion_df['Heading'] = np.arctan2(fusion_df['SpeedX'], fusion_df['SpeedY']) * 180. / np.pi
#     fusion_df['Speed'] = np.sqrt(fusion_df['SpeedX']**2 + fusion_df['SpeedY']**2)
#     fusion_df['ErrorStrength'] = 1. / np.sqrt(fusion_df['X_var'] + fusion_df['Y_var'])
#     fusion_df['ErrorStrength'] = np.clip(
#         fusion_df['ErrorStrength'] / fusion_df['ErrorStrength'].max(), 0.1, 1.0)
#     return fusion_df


# def update_kf_matrices(ith_kf, ith_mm, ith_om):
#     """Update KF matrices from  motion model and observation model"""
#     ith_kf.F = ith_mm.F.copy()
#     ith_kf.Q = ith_mm.Q.copy()
#     ith_kf.H = ith_om.H.copy()
#     ith_kf.R = ith_om.R.copy()
#     ith_kf.state_names = ith_mm.state_names
#     ith_kf.dt_tol = 0.005


# def animate_this_truth(idf, iax, clr, box_fill=True, t_buffer=0.02):
#     """Animate this dataframe"""
#     # eltime = np.around(idf['TimeElapsed'].values, 1)
#     eltime = idf['TimeElapsed'].values
#     xloc = idf['PositionX'].copy().values
#     yloc = idf['PositionY'].copy().values
#     # if 'SDnorthd' in idf.columns:
#     #     xloc_var = idf['SDnorth'].copy().values**2
#     #     yloc_var = idf['SDeast'].copy().values**2
#     #     alphas = 1. / np.sqrt(xloc_var + yloc_var)
#     #     alphas /= np.amax(alphas)
#     #     alphas = np.clip(alphas, 0.9, 1.0)
#     # else:
#     alphas = None
#     length = idf['Length'].values if 'Yaw' in idf.columns else None
#     width = idf['Width'].values if 'Yaw' in idf.columns else None
#     angle = idf['Yaw'].values if 'Yaw' in idf.columns else None
#     return animate_this_source(iax, eltime, xloc, yloc, angle,
#                                width, length, None, clr,
#                                box_fill, alphas, t_buffer)


# def multisensor_fusion_engine(
#     sensor_data_df: pd.DataFrame,
#     kf_base: Dict,
#     sensor_obs_models: Dict,
#     start_state_stds: List[float],
#     loglik_threshold: float,
#     lifespan_after_last_update: float = 10.,
#     min_lifespan: float = 1.,
#     association_ignore: List[int] | None = None,
#     object_lengths: Dict[str, float] | None = None,
#     verbose: bool = False
# ):
#     """ IPC fusion engine"""
#     print('---IPC Multi-sensor Fusion Engine')
#     print(f'Got {sensor_data_df.shape[0]} sensor detections..')
#     print(f'Objects tracked for < {min_lifespan} s  = spurious detections')
#     print(f'Path predicted for {lifespan_after_last_update} s before killing')
#     start_time = time.time()
#     object_list = {}
#     object_list.clear()
#     object_id = 0
#     for _, row in sensor_data_df.iterrows():
#         this_sensor = row['SensorID']
#         this_time = np.around(row['TimeElapsed'], 3)
#         this_obs = np.asarray(row['ObjectList'])
#         if verbose:
#             print(f'--Incoming at {this_time} from sensor {this_sensor}')
#         om = sensor_obs_models[this_sensor]
#         ignore_these_obs = []
#         if association_ignore is not None:
#             ignore_these_obs = association_ignore[this_sensor]
#         loglik = {}
#         dead_object_ids = []
#         loglik.clear()
#         # get loklik of exisiting vehicles
#         for this_id, this_kf in object_list.items():
#             if this_time - this_kf.last_update_at < lifespan_after_last_update:
#                 this_kf.forecast_upto(this_time)
#                 this_kf.H = om.H.copy()
#                 this_kf.R = om.R.copy()
#                 loglik[this_id] = this_kf.get_loglik_of_obs(
#                     y_obs=this_obs.copy(),
#                     ignore_obs_inds=ignore_these_obs
#                 )
#             else:
#                 this_kf.forecast_upto(
#                     this_kf.last_update_at + lifespan_after_last_update)
#                 ltime = this_kf.last_update_at - this_kf.get_time_elapsed()[0]
#                 if ltime < min_lifespan:  # get rid of short tracked objects
#                     dead_object_ids.append(this_id)
#                 # print(f'{this_id} at {this_time},{this_kf.last_update_at}')

#         # delete dead objects
#         for this_id in dead_object_ids:
#             print(f'Killed  object {this_id} at {np.around(this_time,1)} s')
#             del object_list[this_id]
#         if verbose:
#             print('  Logliks: ', [np.around(v, 1) for k, v in loglik.items()])
#             print('  Updates: ', [np.around(v.last_update_at, 2)
#                   for k, v in object_list.items()])

#         # figure our objects that this data may be associated with
#         probable_objects = {}
#         for k, v in loglik.items():
#             if v >= loglik_threshold:
#                 probable_objects[k] = v

#         if bool(probable_objects):
#             found_this_object = max(loglik, key=loglik.get)
#             object_list[found_this_object].obs = this_obs
#             object_list[found_this_object].update()
#         else:
#             print(f'Started object {object_id} at {np.around(this_time,1)} s')
#             object_list[object_id] = deepcopy(kf_base)
#             object_list[object_id].id = object_id
#             object_list[object_id].H = om.H.copy()
#             object_list[object_id].R = om.R.copy()
#             start_state_mean = om.H.T @ this_obs
#             if object_lengths is not None:
#                 len_idx = kf_base.state_names.index('Length')
#                 start_state_mean[len_idx] = object_lengths['Vehicle']
#             object_list[object_id].initiate_state(
#                 this_time,
#                 start_state_mean,
#                 np.diag(start_state_stds**2)
#             )
#             object_id += 1

#     # at the end of tracking
#     dead_object_ids = []
#     for this_id, this_kf in object_list.items():
#         span_time = this_time - this_kf.get_time_elapsed()[0]
#         if span_time < min_lifespan:
#             dead_object_ids.append(this_id)
#     for this_id in dead_object_ids:
#         print(f'Killed  object {this_id} at {np.around(this_time,2)} s')
#         del object_list[this_id]

#     # post processing
#     dfs = [object_list[ix].df_filter for ix in list(object_list.keys())]
#     fusion_df = pd.concat(dfs)
#     fusion_df['ErrorStrength'] = 1. / \
#         np.sqrt(fusion_df['X_var'] + fusion_df['Y_var'])
#     fusion_df['ErrorStrength'] = np.clip(
#         fusion_df['ErrorStrength'] / fusion_df['ErrorStrength'].max(), 0.1, 0.75)
#     print(f'Found a total of {fusion_df.ObjectId.nunique()} object(s)')
#     run_time = np.around(((time.time() - start_time)) / 60., 2)
#     print(f'---took {run_time} mins\n', flush=True)
#     return fusion_df
