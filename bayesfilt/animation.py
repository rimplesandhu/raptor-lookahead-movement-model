""" Functions related to animating the scene """

from typing import Optional, Union
import math
from numpy import ndarray
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as patches
from shapely.geometry import Point


def animate_this_dataframe(ics, idf, ifig, iax, tlist, clr):
    """Animate this dataframe"""
    eltime = idf['time_elapsed'].values
    xloc = idf['x_coord'].copy().values
    yloc = idf['y_coord'].copy().values
    length = idf['length'].copy().values
    width = idf['width'].copy().values
    col_name = ['zone', 'orientation']
    object_ids = None
    if 'object_id' in idf.columns:
        object_ids = idf['object_id'].copy().tolist()
    col_name = ['zone', 'orientation']
    zone_pair = ics.df.loc[ics.df['orientation'].notna(), col_name].values
    if 'yaw' in idf.columns:
        angle = idf['yaw'].values
    else:
        angle_raw = np.arctan2(idf['x_vel'], idf['y_vel']) * 180. / np.pi
        angle = angle_raw.copy().values
        bool_speed = np.sqrt(idf['x_vel']**2 + idf['y_vel']**2) < 1.
        angle[np.where(bool_speed)[0]] = np.nan
        for izone, iangle in zone_pair:
            ibool = [izone.contains(Point(ix, iy))
                     for ix, iy in zip(xloc, yloc)]
            angle[np.where(bool_speed & ibool)[0]] = iangle
    return animate_this_source(ifig, iax, tlist, eltime, xloc, yloc, -angle,
                               width, length, object_ids, clr,
                               interval=50, repeat=False)


def animate_this_source(
    fig,
    ax,
    tlist: ndarray,
    etime: ndarray,
    xloc: ndarray,
    yloc: ndarray,
    yaw: Optional[ndarray] = None,
    width: Union[ndarray, float] = 1.,
    length: Union[ndarray, float] = 1.,
    object_ids: Optional[ndarray] = None,
    obj_color: str = 'r',
    t_buffer: float = 0.001,
    box_fill: bool = True,
    **kwargs
):
    """ Animates the data source """
    assert etime.size == xloc.size, 'Size mismatch b/w etime and xloc!'
    assert xloc.size == yloc.size, 'Size mismatch b/w xloc and yloc!'
    boxes = []
    texts = []
    small_width = 2.5
    max_num_of_objects = 100
    if yaw is not None:
        yaw = yaw.ravel() if yaw.ndim == 1 else yaw
        width = width * \
            np.ones_like(yaw) if isinstance(width, float) else width
        length = length * \
            np.ones_like(yaw) if isinstance(length, float) else length
        assert yloc.size == yaw.size, 'Size mismatch b/w locs and yaw!'
        assert yaw.size == length.size, 'Size mismatch b/w locs and length!'
        assert width.size == length.size, 'Size mismatch b/w locs and width!'
        for _ in range(max_num_of_objects):
            ibox = patches.FancyBboxPatch((0, 0), 0., 0., fill=box_fill,
                                          ec=None, fc=obj_color,
                                          alpha=0.8, lw=0.,
                                          boxstyle='round',
                                          linewidth=0.9)
            ax.add_patch(ibox)
            boxes.append(ibox)
            if object_ids is not None:
                itext = ax.text(0., 0., '', horizontalalignment='center',
                                verticalalignment='center',
                                color='w', fontsize=5)

                texts.append(itext)

        width = width * \
            np.ones_like(xloc) if isinstance(width, float) else width
        length = length * \
            np.ones_like(xloc) if isinstance(length, float) else length
    msize = 5. if yaw is None else 1.
    points_plot, = ax.plot([], 'o' + obj_color,
                           alpha=0.5, markersize=msize)
    rot_start = ax.transData

    def animate_func(itime: float):
        tbool = np.logical_and(etime >= itime - t_buffer,
                               etime <= itime + t_buffer)
        inds = np.where(tbool)[0]
        points_plot.set_data(xloc[inds], yloc[inds])
        if yaw is not None:
            for i, indx in enumerate(inds):
                iangle = yaw[indx]
                boxes[i].set_boxstyle('round', rounding_size=1.)
                coords = rot_start.transform((xloc[indx], yloc[indx]))
                if not math.isnan(iangle):
                    boxes[i].set_x(xloc[indx] - width[indx] / 2)
                    boxes[i].set_y(yloc[indx] - length[indx] / 2)
                    boxes[i].set_width(width[indx])
                    boxes[i].set_height(length[indx])
                    rot_this = mpl.transforms.Affine2D().rotate_deg_around(
                        coords[0], coords[1], iangle)
                    boxes[i].set_transform(rot_start + rot_this)
                else:
                    boxes[i].set_x(xloc[indx] - small_width / 2)
                    boxes[i].set_y(yloc[indx] - small_width / 2)
                    boxes[i].set_width(small_width)
                    boxes[i].set_height(small_width)
                    rot_this = mpl.transforms.Affine2D().rotate_deg_around(
                        coords[0], coords[1], 0.)
                    boxes[i].set_transform(rot_start + rot_this)
                if object_ids is not None:
                    obj_id = object_ids[indx]
                    texts[i].set_text(str(obj_id))
                    texts[i].set_x(xloc[indx])
                    texts[i].set_y(yloc[indx])
            for j in range(len(inds), max_num_of_objects):
                #boxes[j].set_boxstyle('round', pad=0)
                boxes[j].set_x(0.)
                boxes[j].set_y(0.)
                boxes[j].set_width(0.)
                boxes[j].set_height(0.)
                if object_ids is not None:
                    texts[j].set_text('')
        return points_plot, boxes,
    return animation.FuncAnimation(fig, animate_func, tlist, **kwargs)
