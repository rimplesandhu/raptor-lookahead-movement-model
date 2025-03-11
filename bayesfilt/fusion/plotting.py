"""Functions useful for plotting"""
import sys
from pathlib import Path
import glob
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib.patches import Patch
from .utils import run_loop
from .traffic_sensor import TrafficSensor

cb_clrs = ['#377eb8', '#ff7f00', '#4daf4a',
           '#f781bf', '#a65628', '#984ea3',
           '#999999', '#e41a1c', '#dede00']


def intersection_plot_initiator(
    figsize: tuple[float, float] = (6, 6),
    xlim: tuple[float, float] = [-100, 100],
    ylim: tuple[float, float] = [-100, 100],
    left_str: str = 'Colorado Springs',
    close_fig: bool = False,
    clr: str = 'lavender',
):
    """Initialize the plot for creating animations"""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.set_aspect('equal')
    ax.axis(False)
    # ax.tick_params(axis="y", direction="in", pad=-30)
    # ax.tick_params(axis="x", direction="in", pad=-20)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # xticks = ax.get_xticks()
    # xticks = xticks[1:-1]
    # ax.set_xticks(xticks)
    # yticks = ax.get_yticks()
    # yticks = yticks[1:-1]
    # ax.set_yticks(yticks)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor(clr)
    ax.text(0.01, 0.99, str(left_str), ha='left', va='top',
            color='k', transform=ax.transAxes,
            bbox=dict(fc=clr, ec=clr))
    if close_fig:
        plt.close(fig)
    return fig, ax


def intersection_legend_creator(ax, sensors: list[TrafficSensor]):
    list_of_patches = []
    for isensor in sensors:
        ipatch = Patch(
            color=isensor.clr,
            label=isensor.name
        )
        list_of_patches.append(ipatch)
    lg = ax.legend(
        handles=list_of_patches,
        loc=1,
        ncols=1,
        borderpad=0.2,
        # handlelength=1.5,
        fontsize=9,
    )
    lg.get_frame().set_alpha(0)
    return lg


def add_colored_legend(ax, idict):
    """add legend"""
    for ix, (iname, iclr) in enumerate(idict.items()):
        ax.text(.0, 0.01 + ix * 0.05, iname,
                fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle="round", fc=iclr, ec='none', alpha=0.25))


def create_video_from_frames(
    image_dir: str,
    fps: int,
    fpath: str,
    delete_images_after: bool = False
):
    """Creates/saves video from images saved as png files"""
    # print(image_dir, filename)
    # get the images
    images = [img for img in os.listdir(image_dir) if img.endswith(".png")]
    print(f'Creating video using {len(images)} frames at {fps} FPS..',
          end="", flush=True)
    images.sort()
    frame = cv2.imread(os.path.join(image_dir, images[0]))

    # get the size of frame
    height, width, _ = frame.shape
    fncc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(fpath, fncc, fps, (width, height))

    # add images to video
    looper = tqdm(
        iterable=images,
        total=len(images),
        ncols=80,
        leave=True,
        file=sys.stdout,
        desc='AnimVideo',
        disable=False
    )
    for image in looper:
        video.write(cv2.imread(os.path.join(image_dir, image)))
        if delete_images_after:
            os.remove(os.path.join(image_dir, image))

    # finish and wrap up
    cv2.destroyAllWindows()
    video.release()
    # print('done')


def merge_frames(dir1: str, dir2: str, out_dir: str) -> None:
    """Merge frames side by side"""
    # print(f'Merging frames togethor..', end="", flush=True)
    flist1 = sorted(glob.glob(os.path.join(dir1, "*.png")))
    flist2 = sorted(glob.glob(os.path.join(dir2, "*.png")))
    assert len(flist1) == len(flist2), 'Need same number of images!'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    looper = tqdm(
        iterable=zip(flist1, flist2),
        total=len(flist1),
        position=0,
        ncols=80,
        leave=True,
        file=sys.stdout,
        desc='AnimMerge',
        disable=False
    )
    for file1, file2 in looper:
        image1 = cv2.imread(file1)
        image2 = cv2.imread(file2)
        vis = np.concatenate((image1, image2), axis=1)
        fout = os.path.join(out_dir, file1.split('/')[-1])
        cv2.imwrite(fout, vis)
    # print('done', flush=True)


def get_frame_fname(ix):
    """Returns filename for a given frame id"""
    return f'frame.{str(ix).zfill(6)}.png'
