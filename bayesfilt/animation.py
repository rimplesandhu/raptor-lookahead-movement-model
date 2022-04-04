""" Functions related to animating the scene """

from typing import Optional, Union
from numpy import ndarray
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patches as patches


def animate2d_this_source(
    fig,
    ax,
    etime: ndarray,
    xloc: ndarray,
    yloc: ndarray,
    yaw: Optional[ndarray] = None,
    width: Union[ndarray, float] = 1.,
    length: Union[ndarray, float] = 1.,
    max_num_of_objects: int = 100,
    obj_color: str = 'r',
    t_buffer: float = 0.01,
    box_fill: bool = False,
    **kwargs

):
    """ Animates the data source """

    # check  arrays are compatible
    # etime = etime.ravel() if etime.ndim == 1 else etime
    # xloc = xloc.ravel() if xloc.ndim == 1 else xloc
    # yloc = xloc.ravel() if yloc.ndim == 1 else yloc

    assert etime.size == xloc.size, 'Size mismatch b/w etime and xloc!'
    assert xloc.size == yloc.size, 'Size mismatch b/w xloc and yloc!'

    # initiate the rect boxes
    boxes = []
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
            ibox = patches.Rectangle((0, 0), 0., 0., fill=box_fill,
                                     edgecolor=obj_color,
                                     joinstyle='round')
            ax.add_patch(ibox)
            boxes.append(ibox)
        width = width * \
            np.ones_like(xloc) if isinstance(width, float) else width
        length = length * \
            np.ones_like(xloc) if isinstance(length, float) else length

    # draw the points
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
                southwest = (xloc[indx] - width[indx] / 2,
                             yloc[indx] - length[indx] / 2)
                boxes[i].set_xy(southwest)
                boxes[i].set_width(width[indx])
                boxes[i].set_height(length[indx])
                center = (xloc[indx], yloc[indx])
                coords = rot_start.transform(center)
                rot_this = mpl.transforms.Affine2D().rotate_deg_around(
                    coords[0], coords[1], yaw[indx])
                boxes[i].set_transform(rot_start + rot_this)
            for j in range(len(inds), max_num_of_objects):
                boxes[j].set_xy((0., 0.))
                boxes[j].set_width(0.)
                boxes[j].set_height(0.)
        return points_plot, boxes,
    start_time = np.floor(np.amin(etime))
    end_time = np.ceil(np.amax(etime))
    dtlist = np.diff(etime)
    time_interval = np.round(np.amin(dtlist[dtlist > 0.]) * 100) / 100.
    print(f'From {start_time} to {end_time} seconds at {time_interval}')
    num_frames = int(np.round((end_time - start_time) / time_interval))
    tlist = np.linspace(start_time, end_time, num_frames + 1)
    return animation.FuncAnimation(fig, animate_func, tlist, **kwargs)
