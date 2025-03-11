#!/usr/bin/env python
"""Traffic sensor class"""

# pylint: disable=invalid-name
from __future__ import annotations
import glob
import sys
import os
from copy import deepcopy
import warnings
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from numpy import ndarray
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, Rectangle, Patch, FancyBboxPatch, BoxStyle
import cartopy.crs as ccrs
from tqdm import tqdm

from .utils import run_loop, rotate_clockwise
warnings.simplefilter('ignore')

# GEO_CRS = ccrs.CRS('EPSG:4326')
# PROJ_CRS = ccrs.UTM(13)  # 'EPSG:3857'
CLRS10 = iter(list(plt.cm.tab10.colors))


class TrafficSensor:
    """Base class for sensors"""

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = 'Sensor0',
        clr: str = 'b',
        verbose: bool = False,
        objectid_colname: str = 'ObjectId',
        time_colname: str = 'Time',
        x_colname: str = 'PositionX',
        y_colname: str = 'PositionY'
    ):

        # basics
        self.name = name
        self.clr = clr
        self.verbose = verbose

        # dataframe contianing data
        self.printit(f'Initiating traffic sensor {name}..')
        self.df = df
        self.idcol = objectid_colname
        self.tcol = time_colname
        self.xcol = x_colname
        self.ycol = y_colname
        self.check_column_names([self.idcol, self.tcol, self.xcol, self.ycol])

        # time info
        self.df[self.tcol] = pd.to_datetime(self.df[self.tcol]).dt.tz_localize(None)
        #self.df[self.tcol] = self.df[self.tcol].dt.tz_localize(None)
        self.printit(f'starts at {self.df[self.tcol].min()}')
        self.printit(f'  ends at {self.df[self.tcol].max()}')

        # catogorize ids
        codes, _ = pd.factorize(self.df[self.idcol])
        self.df.rename(columns={self.idcol: f'{self.idcol}_RAW'})
        self.df[self.idcol] = codes
        n_objs = self.df[self.idcol].nunique()
        self.printit(f'{len(self.df):,} entries from {n_objs:,} objects')

        # other info
        self.tecol = 'TimeElapsed'
        self.cnames = [self.idcol, self.tecol, self.xcol, self.ycol]
        self.df['IgnoreThese'] = False
        self.sensor_loc = (0., 0.)
        self.sensor_loc_tmp = [0., 0.]
        self.width_col = None
        self.length_col = None
        self.heading_col = None
        self.alpha_col = None
        self.show_ids: bool = False
        self.dff = None


    def __repr__(self):
        min_t = self.df[self.tcol].min().time().strftime("%H:%M:%S")
        max_t = self.df[self.tcol].max().time().strftime("%H:%M:%S")
        return (f'{self.__class__.__name__}('
                f'name={self.name}, '
                f'size={self.df.shape[0]:,}, '
                f'trange={min_t}-{max_t})'
                )

    def printit(self, istr: str):
        """Print command"""
        print(self.name + ':', istr, flush=True)

    def check_column_names(self, cnames: list[str]):
        """check if column exists in data"""
        cnames = cnames if isinstance(cnames, list) else [cnames]
        for cname in cnames:
            if cname not in self.df.columns:
                raise ValueError(f'{cname} not found it dataframe!')

    def sort_df_by(self, cnames: list[str]):
        """sort the pandas dataframe"""
        self.df.set_index(cnames, inplace=True, drop=True)
        self.df.sort_index(inplace=True)
        self.df.reset_index(drop=False, inplace=True)

    def compute_timeelapsed(self, base_time: datetime, round_to: str = '100ms') -> None:
        """add time elapsed and time diff columns"""
        self.df[self.tcol] = self.df[self.tcol].dt.round(round_to)
        self.df[self.tecol] = self.df[self.tcol] - base_time
        self.df[self.tecol] = self.df[self.tecol].dt.total_seconds()
        self.df[self.tecol] = self.df[self.tecol].round(decimals=3)
        min_time = np.around(self.df[self.tecol].min(), 3)
        max_time = np.around(self.df[self.tecol].max(), 3)
        self.printit(f'Time range: {min_time}-{max_time} sec')

    def compute_speed_from_position(self):
        """Computes speed info from position data"""
        
        # figure out the start of track
        self.sort_df_by([self.idcol, self.tcol])
        track_start = self.df[self.idcol].diff().astype(bool).fillna(True)
        
        # compute speeds
        xscol = 'SpeedX_DERIVED'
        self.df[xscol] = self.df[self.xcol].diff()
        yscol = 'SpeedY_DERIVED'
        self.df[yscol] = self.df[self.ycol].diff()

        # finish
        for icol in [xscol, yscol]:
            self.df[icol] /= self.df.TimeElapsed.diff()
            self.df.loc[track_start, icol] = np.nan
            self.df[icol].bfill(inplace=True)
        self.df['Speed_DERIVED'] = self.df[xscol]**2+self.df[yscol]**2
        self.df['Speed_DERIVED'] = self.df['Speed_DERIVED'].apply(np.sqrt)

    def transform(
        self,
        angle_deg: float,
        xdist: float = 0.,
        ydist: float = 0.,
        heading_col: str | None = None,
        speedx_col: str | None = None,
        speedy_col: str | None = None

    ) -> None:
        """Rotate the telelmetry data by some angle"""

        # position
        xnew, ynew = rotate_clockwise(
            self.df[self.xcol],
            self.df[self.ycol],
            angle_deg=angle_deg
        )
        self.df[f'{self.xcol}_Transformed'] = xnew + xdist
        self.df[f'{self.ycol}_Transformed'] = ynew + ydist
        self.sensor_loc_tmp[0] = self.sensor_loc[0] + xdist
        self.sensor_loc_tmp[1] = self.sensor_loc[1] + ydist

        # heading
        if heading_col in self.df.columns:
            if not self.df[heading_col].between(-180, 180).all():
                raise ValueError(f'Heading data in column {heading_col}'
                                 ' is not between (-180,180]')
            cname = f'{heading_col}_Transformed'
            self.df[cname] = self.df[heading_col] - angle_deg
            self.df.loc[self.df[cname] > 180., cname] -= 360.
            self.df.loc[self.df[cname] <= -180., cname] += 360.
        else:
            self.printit('Heading not considered for coord rotation.')

        # speed
        if (speedx_col in self.df.columns) and (speedy_col in self.df.columns):
            xnew, ynew = rotate_clockwise(
                xvals=self.df[speedx_col],
                yvals=self.df[speedy_col],
                angle_deg=angle_deg
            )
            self.df[f'{speedx_col}_Transformed'] = xnew
            self.df[f'{speedy_col}_Transformed'] = ynew
        else:
            self.printit('SpeedX/SpeedY not considered for coord rotation.')

    def get_lifespan_by_object_id(self, objectid_col: str):
        """returns adataframe with object durations"""
        max_time = self.df.groupby(objectid_col)[self.tecol].max()
        min_time = self.df.groupby(objectid_col)[self.tecol].min()
        outdf = pd.DataFrame({
            'StartTime': min_time,
            'EndTime': max_time,
            'Duration': max_time-min_time
            # 'SpeedMean': groupby['Speed'].mean(),
            # 'SpeedRange': groupby['Speed'].max() - groupby['Speed'].min(),
        })
        outdf.sort_values(by='Duration', inplace=True, ascending=False)
        return outdf

    def replace_original_with_transformed(self):
        """replace original data with transformed"""
        self.sensor_loc = deepcopy(self.sensor_loc_tmp)
        for cname in self.df.columns:
            if '_Transformed' in cname:
                rname = cname.rsplit('_', 1)[0]
                self.df.rename(
                    mapper={rname: f'{rname}tmp'},
                    axis=1,
                    inplace=True
                )
                self.df.rename(
                    mapper={cname: rname},
                    axis=1,
                    inplace=True
                )
                self.df.drop(
                    columns=[f'{rname}tmp'],
                    inplace=True
                )

    def ignore_these_detections(self, condition: pd.Series):
        """Bool Series for ignoring data"""
        if not pd.api.types.is_bool_dtype(condition):
            raise ValueError('Need Bool pandas Series!')
        cname = 'IgnoreThese'
        self.df[cname] = (self.df[cname]) | (condition)
        perc = np.around(self.df[cname].sum()*100/len(self.df), 2)
        if self.verbose:
            self.printit(f'{perc}% data ignored')

    def update_dff(self):
        """Update dff dataframe needed for fusion"""
        iname = 'IgnoreThese'
        if iname not in self.df.columns:
            self.df[iname] = False
        self.dff = self.df.loc[~self.df[iname], self.cnames].copy()
        self.dff.set_index([self.idcol, self.tecol], inplace=True, drop=True)
        self.dff.sort_index(inplace=True)
        self.dff.reset_index(drop=False, inplace=True)

    def setup_drawing(
        self,
        width_col: str | None = None,
        length_col: str | None = None,
        headingdeg_col: str | None = None,
        alpha_col: str | None = None,
        show_ids: bool = False
    ):
        """set up drawing of objects on a frame"""
        if width_col in self.df.columns:
            self.check_column_names([length_col, headingdeg_col])
            self.printit('Width/Length/Heading Found. Rectangular objects!')
            self.width_col = width_col
            self.length_col = length_col
            self.heading_col = headingdeg_col
            self.cnames += [self.width_col, self.length_col, self.heading_col]
            self.cnames = list(set(self.cnames))
        else:
            self.printit('No heading/width/length provided. Circular objects!')

        if alpha_col is not None:
            self.check_column_names(alpha_col)
            self.alpha_col = alpha_col
            self.cnames += [self.alpha_col]
            self.cnames = list(set(self.cnames))

        self.show_ids = show_ids

    def get_object_list(
            self,
            varnames: List[str],
            timeelapsed_range=None
    ) -> pd.DataFrame:
        """Returns object list"""
        self.check_column_names(varnames)
        if timeelapsed_range is not None:
            assert 'TimeElapsed' in self.df.columns, 'run compute_timeelapsed() first!'
            ibool = self.df['TimeElapsed'].between(*timeelapsed_range)
        else:
            ibool = self.df.index

        ibool = (ibool) & (~self.df['IgnoreThese'])
        out_df = pd.DataFrame({
            'Time': self.df.loc[ibool, 'Time'].values,
            'TimeElapsed': self.df.loc[ibool, 'TimeElapsed'].values,
            'Data': self.df.loc[ibool, varnames].values.tolist()
        })
        out_df['Sensor'] = pd.Series(
            [self.name]*out_df.shape[0], dtype='category')
        out_df.set_index(['TimeElapsed', 'Sensor'], inplace=True)
        out_df.sort_index(inplace=True)
        out_df.reset_index(drop=False, inplace=True)
        return out_df

    def draw_frame(
        self,
        ax: mpl.axes.Axes,
        at_time_elapsed: float,
        time_padding: float
    ) -> None:
        """Draws frame at this time"""

        # get detected objets within the time frame
        time_low = at_time_elapsed - time_padding
        time_upp = at_time_elapsed + time_padding
        tbool = self.dff[self.tecol].between(time_low, time_upp)
        # dfshort = self.df.loc[(tbool) | (fbool), :]
        dfshort = self.dff[(tbool)].copy()
        # dbool = ~dfshort.duplicated(subset=self.idcol, keep='last')
        # dfshort = dfshort.loc[dbool, :]
        dfshort.drop_duplicates([self.idcol], keep='last', inplace=True)

        # iter and draw all the detected objects
        for _, irow in dfshort.iterrows():
            centroid = (irow[self.xcol], irow[self.ycol])
            if self.width_col is not None:
                width = irow[self.width_col]
                length = irow[self.length_col]
                lowerleft_loc = (
                    centroid[0] - width / 2.,
                    centroid[1] - length / 2.
                )

                angle_deg = irow[self.heading_col]
                angle_deg = (angle_deg + 90) % 360.
                rot_transform = mpl.transforms.Affine2D().rotate_deg_around(
                    centroid[0],
                    centroid[1],
                    angle_deg
                ) + ax.transData
                ibox = FancyBboxPatch(
                    xy=lowerleft_loc,
                    width=width,
                    height=length,
                    boxstyle=BoxStyle('round', rounding_size=1.),
                    transform=rot_transform,
                    ec='none',
                    fc=self.clr,
                    alpha=0.75 if self.alpha_col is None else irow[self.alpha_col]
                )

                # ibox = Rectangle(
                #     xy=lowerleft_loc,
                #     width=width,
                #     height=length,
                #     angle=angle_deg,
                #     rotation_point='center',
                #     ec='none',
                #     fc=self.clr,
                #     alpha=0.75 if self.alpha_col is None else irow[self.alpha_col]
                # )
            else:
                ibox = Circle(
                    xy=centroid,
                    radius=1.5,
                    ec='none',
                    fc=self.clr,
                    alpha=0.75 if self.alpha_col is None else irow[self.alpha_col]
                )
            ax.add_patch(ibox)
            if self.show_ids:
                ax.text(
                    centroid[0],
                    centroid[1],
                    str(int(irow[self.idcol]))[-2:],
                    fontsize=3.5,
                    ha='center',
                    va='center',
                    color='w'
                )


def create_frames_from_sensors(
    list_of_sensors: list[TrafficSensor],
    list_of_etimes: list[float],
    out_dir: str,
    fig_init_fn: callable,
    ncores: int = 1,
    dpi: int = 200,
):
    """Create animation from a list of dataframes"""
    # initiate
    # istr = f'{round(list_of_etimes[0], 3)}-{round(list_of_etimes[-1], 3)}'
    # print(f'Creating frames within time {istr}..')

    # remove existing files from the dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filelist = glob.glob(os.path.join(out_dir, "*.png"))
    _ = [os.remove(f) for f in filelist]

    # new set of files
    fnames = [f'f{str(ix).zfill(6)}.png' for ix in range(len(list_of_etimes))]
    fpaths = [os.path.join(out_dir, ix) for ix in fnames]
    figures = [fig_init_fn() for _ in range(len(list_of_etimes))]
    _ = [plt.close(fig) for fig, _ in figures]
    loop_tuple = [(ix[1], iy) for ix, iy in zip(figures, list_of_etimes)]

    # consider data only within the axis limits
    _, ax = figures[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    pad = 10.

    # run the loop
    for isensor in list_of_sensors:

        # ignore detections outside axis bounds
        xbool = isensor.df[isensor.xcol].between(xmin+pad, xmax-pad)
        ybool = isensor.df[isensor.ycol].between(ymin+pad, ymax-pad)
        isensor.ignore_these_detections(condition=~(xbool & ybool))
        isensor.update_dff()

        def draw_frame_fn(ituple):
            return isensor.draw_frame(
                ax=ituple[0],
                at_time_elapsed=ituple[1],
                time_padding=(list_of_etimes[1] - list_of_etimes[0])/2.,
            )
        _ = run_loop(
            func=draw_frame_fn,
            tqdm_pbar=tqdm(
                iterable=loop_tuple,
                total=len(loop_tuple),
                desc=f'{isensor.name}-Draw',
                leave=True,
                ncols=80,
                file=sys.stdout,
                disable=False
            ),
            ncores=ncores
        )

    # create legend manually
    list_of_patches = []
    for isensor in list_of_sensors:
        ipatch = Patch(
            color=isensor.clr,
            label=isensor.name
        )
        list_of_patches.append(ipatch)

    # finish and save

    def _save_frames(ix):
        fpath, (fig, ax) = ix
        _ = ax.legend(
            handles=list_of_patches,
            loc=1,
            ncols=2,
            frameon=False,
            # handlelength=1.5,
            fontsize=9,
        )
        fig.savefig(fpath, dpi=dpi)

    _ = run_loop(
        func=_save_frames,
        tqdm_pbar=tqdm(
            iterable=list(zip(fpaths, figures)),
            total=len(list(zip(fpaths, figures))),
            desc=f'SaveAllSensors',
            leave=True,
            ncols=80,
            file=sys.stdout,
            disable=False
        ),
        ncores=ncores
    )

# class TrafficSensor:
#     """Base class for defining a traffic sensor that produces an object list"""

#     def __init__(
#         self,
#         name: str,
#         object_id: list[int],
#         tdata: list[datetime],
#         xdata: list[float],
#         ydata: list[float],
#         speed: list[float] | None = None,
#         speed_x: list[float] | None = None,
#         speed_y: list[float] | None = None,
#         heading_deg: list[float] | None = None,
#         width: list[float] | None = None,
#         length: list[float] | None = None,
#         loc_lonlat: Tuple[float, float] | None = None,
#         uncertainty: list[float] | None = None,
#         clr: str = 'b',
#         verbose: bool = False,
#         show_ids: bool = True,
#         **kwargs
#     ):

#         # check compatibility of inputs
#         assert len(tdata) == len(xdata), 'tdata/xdata size mismatch!'
#         assert len(xdata) == len(ydata), 'xdata/ydata size mismatch!'

#         # basic info
#         self.name = name
#         self.clr = clr
#         self.verbose = verbose
#         self.show_ids = show_ids

    # # some checks
    # # self.printit(f'Initiating...')
    # self.df = pd.DataFrame(dict(
    #     Time=pd.to_datetime(tdata),
    #     PositionX=xdata,
    #     PositionY=ydata,
    #     Distance=np.sqrt(xdata**2+ydata**2),
    #     **kwargs
    # ))
    # self.df['Time'] = self.df['Time'].dt.tz_localize(None)
    # # self.df['Time'] = self.df['Time'].dt.round('1ms')

    # # sensor location, compute lon/lat positions
    # self.loc_sensor_lonlat = loc_lonlat
    # self.compute_lonlats()

    # # deal with not integer object id's
    # assert len(xdata) == len(object_id), 'xdata/object_id size mismatch!'
    # self.df['ObjectId'] = object_id
    # codes, _ = pd.factorize(self.df['ObjectId'])
    # self.df['ObjectId'] = codes
    # # self.df['ObjectId'] = self.df['ObjectId'].astype(np.int32)
    # self.df['ObjectId'] -= self.df['ObjectId'].min()
    # nobj = self.df.ObjectId.nunique()
    # self.printit(f'{self.df.shape[0]} detections from {nobj} objects')

    #     # alpha component
    #     if uncertainty is not None:
    #         self.df['Uncertainty'] = uncertainty
    #         self.df['UncertaintyNorm'] = self.df['Uncertainty'].copy()
    #         self.df['UncertaintyNorm'] -= self.df['UncertaintyNorm'].min()
    #         self.df['UncertaintyNorm'] /= self.df['UncertaintyNorm'].max()

    #     # deal with absense of heading or speedx/speedy data
    #     if heading_deg is not None:
    #         self.printit('Heading (deg) is assumed +ve CCW from x axis!')
    #         # assert np.amax(heading_deg) > 180., 'Heading outside [180,-180)!'
    #         # assert np.amin(heading_deg) < -180., 'Heading outside [180,-180)!'
    #         self.df['Heading'] = heading_deg
    #         self.df.loc[self.df['Heading'] > 180., 'Heading'] -= 360
    #         self.df.loc[self.df['Heading'] <= -180., 'Heading'] += 360
    #         assert speed is not None, 'Need speed info along with heading'
    #         self.df['Speed'] = speed
    #     if speed_x is not None:
    #         assert speed_y is not None, 'Need both speedx and speedy!'
    #         self.df['SpeedX'] = speed_x
    #         self.df['SpeedY'] = speed_y
    #         self.df['Speed'] = np.sqrt(speed_x**2 + speed_y**2)
    #         if heading_deg is None:
    #             self.printit('computing Heading using SpeedX/SpeedY data')
    #             self.df['Heading'] = np.arctan2(speed_y, speed_x)
    #             self.df['Heading'] = np.degrees(self.df['Heading'])
    #     else:
    #         assert speed_y is None, 'Need both speedx and speedy!'
    #         if heading_deg is not None:
    #             assert speed is not None, 'Need speed info along with heading'
    #             self.df['Speed'] = speed
    #             self.printit('computing SpeedX/SpeedY using Speed/Heading')
    #             self.df['SpeedX'] = speed *
    #             np.cos(np.radians(self.df['Heading']))
    #             self.df['SpeedY'] = speed *
    #             np.sin(np.radians(self.df['Heading']))

    #     # deal with width and length data
    #     if width is None:
    #         self.printit(f'No object width data supplied!')
    #     else:
    #         assert len(width) == len(xdata), 'width/xdata length mismatch!'
    #         self.df['Width'] = width
    #     if length is None:
    #         self.printit(f'No object length data supplied!')
    #     else:
    #         self.df['Length'] = length

    #     # data types for the dataframes
    #     self.df.set_index(['ObjectId', 'Time'], inplace=True, drop=True)
    #     self.df.sort_index(inplace=True)
    #     self.df.reset_index(drop=False, inplace=True)
    #     if self.df.memory_usage(index=True).sum()/1024/1024 > 50.:
    #         float64_cols = list(self.df.select_dtypes(include='float64'))
    #         self.df[float64_cols] = self.df[float64_cols].astype('float32')
    #     # self.dfraw = self.df.copy()
    #     self.printit(f'starts at {self.df['Time'].min()}')
    #     self.printit(f'  ends at {self.df['Time'].max()}')

    #     # if heading_deg is not None:
    #     #     self.shape = 'rectangle'
    #     #
    #     # else:
    #     #     self.shape = 'circle'
    #     #     self.boxstyle = patches.BoxStyle('circle', pad=0.3)

    # def compute_speed_from_position(self):
    #     """Compute speed/heading from position"""
    #     track_start = self.df['ObjectId'].diff().astype(bool).fillna(True)
    #     self.df['SpeedX'] = self.df.PositionX.diff()
    #     self.df['SpeedX'] /= self.df.TimeElapsed.diff()
    #     self.df.loc[track_start,
    #                 'SpeedX'] = self.df.loc[track_start.shift(-1), 'SpeedX']

    #     self.df['SpeedY'] = self.df.PositionY.diff()
    #     self.df['SpeedY'] /= self.df.TimeElapsed.diff()
    #     self.df.loc[track_start,
    #                 'SpeedY'] = self.df.loc[track_start.shift(-1), 'SpeedY']

    #     self.df['Heading'] = np.arctan2(self.df.SpeedY, self.df.SpeedX)
    #     self.df['Heading'] = np.degrees(self.df['Heading'])

    # # def compute_curvature(self):
    # #     """Compute curvature"""
    # #     if 'Heading' in self.df.columns:
    # #         self.df['HeadingRate'] = self.df['Heading'].diff().bfill()
    # #     if 'Speed' in self.df.columns:
    # #         speed = self.df['Speed'].clip()
    # #         self.df['Curvature'] = self.df['HeadingRate']/self.df['Speed']

    # # def add_to_df(self, cname: str, data: ndarray):
    # #     """add column to dataframe"""

    # def add(self, isensor: TrafficSensor) -> TrafficSensor:
    #     """add two Traffic Sensor objects"""
    #     _obj1 = deepcopy(self)
    #     _obj2 = deepcopy(isensor)
    #     if not isinstance(isensor, TrafficSensor):
    #         return NotImplemented

    #     if not (sorted(list(_obj1.df.columns)) == sorted(list(_obj2.df.columns))):
    #         raise ValueError('Columns dont match!')

    #     # deal with duplicate object ids and then concat
    #     _obj2.df['ObjectId'] += _obj1.df['ObjectId'].max() + 1
    #     _obj1.df = pd.concat([_obj1.df, _obj2.df])
    #     _obj1.name = f'{_obj1.name}+{_obj2.name}'

    #     # sorting
    #     _obj1.df.set_index(['ObjectId', 'Time'], inplace=True, drop=True)
    #     _obj1.df.sort_index(inplace=True)
    #     _obj1.df.reset_index(drop=False, inplace=True)

    #     return _obj1

    # def __repr__(self) -> str:
    #     """repr"""
    #     cls = self.__class__.__name__
    #     nobj = self.df['ObjectId'].nunique()
    #     return f'{cls}(name={self.name}, n_det={self.df.shape[0]}, n_obj={nobj})'

    # def compute_lonlats(self) -> None:
    #     """Compute latitute and longitude of sensor data"""
    #     if self.loc_sensor_lonlat is not None:
    #         assert len(
    #             self.loc_sensor_lonlat) == 2, 'Need 2 floats in loc_lonlat!'
    #         # get position in meters in proj ref system
    #         loc_xy = PROJ_CRS.transform_points(
    #             src_crs=GEO_CRS,
    #             x=np.array([self.loc_sensor_lonlat[0]]),
    #             y=np.array([self.loc_sensor_lonlat[1]])
    #         )[0]
    #         self.loc_sensor_xy = (loc_xy[0], loc_xy[1])
    #         pos_x = self.df['PositionX'] + loc_xy[0]
    #         pos_y = self.df['PositionY'] + loc_xy[1]

    #         # compute lon and lats
    #         lonlats = GEO_CRS.transform_points(
    #             src_crs=PROJ_CRS,
    #             x=pos_x,
    #             y=pos_y
    #         )
    #         self.df['Longitude'] = lonlats[:, 0]
    #         self.df['Latitude'] = lonlats[:, 1]

    # # def set_origin(self, xloc:float, yloc:)
    # # def add_time_bias(self, time_sec: float):
    # #     """Add seconds to time"""
    # #     self.printit(f'Adding {time_sec} sec to time..')
    # #     self.df['Time'] = self.dfraw['Time'] + timedelta(seconds=time_sec)
    # #     self.printit('Make sure to recompute time elapsed!')
    # #     # if 'TimeElapsed' in self.df.columns:
    # #     #     self.df['TimeElapsed'] += time_sec

    # def fix_heading_for_low_speeds(self, cutoff_speed: float):
    #     """Make sure heading doesnot fluctuate for loww speed vehicles"""
    #     low_speeds = self.df['Speed'] < cutoff_speed
    #     track_start = self.df['ObjectId'].diff().astype(bool).fillna(True)
    #     self.df.loc[(low_speeds) & (~track_start), 'Heading'] = np.nan
    #     self.df['Heading'].ffill(inplace=True)

    # def transform(
    #         self,
    #         angle_deg: float,
    #         xdist: float = 0.,
    #         ydist: float = 0.
    # ) -> None:
    #     """Rotate the data by some angle"""
    #     # heading
    #     if 'Heading' in self.df.columns:
    #         self.df['Heading_Tr'] = self.df['Heading'] - angle_deg
    #         self.df.loc[self.df['Heading_Tr'] > 180., 'Heading_Tr'] -= 360.
    #         self.df.loc[self.df['Heading_Tr'] <= -180., 'Heading_Tr'] += 180.

    #     # position
    #     xnew, ynew = self._rotate_clockwise(
    #         self.df['PositionX'],
    #         self.df['PositionY'],
    #         angle_deg=angle_deg
    #     )
    #     self.df['PositionX_Tr'] = xnew
    #     self.df['PositionY_Tr'] = ynew
    #     self.compute_lonlats()

    #     # speed
    #     if 'SpeedX' in self.df.columns:
    #         xnew, ynew = self._rotate_clockwise(
    #             xvals=self.df['SpeedX'],
    #             yvals=self.df['SpeedY'],
    #             angle_deg=angle_deg
    #         )
    #         self.df['SpeedX_Tr'] = xnew
    #         self.df['SpeedY_Tr'] = ynew

    #     # translate
    #     self.df['PositionX_Tr'] += xdist
    #     self.df['PositionY_Tr'] += ydist
    #     self.loc_sensor_xy = (xdist, ydist)
    #     self.compute_lonlats()

    # @property
    # def df_duration(self):
    #     """returns adataframe with object durations"""
    #     max_time = self.df.groupby('ObjectId')['TimeElapsed'].max()
    #     min_time = self.df.groupby('ObjectId')['TimeElapsed'].min()
    #     outdf = pd.DataFrame({
    #         'StartTime': min_time,
    #         'EndTime': max_time,
    #         'Duration': max_time-min_time
    #         # 'SpeedMean': groupby['Speed'].mean(),
    #         # 'SpeedRange': groupby['Speed'].max() - groupby['Speed'].min(),
    #     })
    #     outdf.sort_values(by='Duration', inplace=True, ascending=False)
    #     return outdf

    # def compute_timeelapsed(self, base_time: datetime, round_to: str = '100ms') -> None:
    #     """add time elapsed and time diff columns"""
    #     self.df['Time'] = self.df['Time'].dt.round(round_to)
    #     self.df['TimeElapsed'] = self.df['Time'] - base_time
    #     self.df['TimeElapsed'] = self.df['TimeElapsed'].dt.total_seconds()
    #     self.df['TimeElapsed'] = self.df['TimeElapsed'].round(decimals=3)
    #     # self.df['TimeDiff'] = self.df['TimeElapsed'].diff().fillna(0.)
    #     min_time = np.around(self.df.TimeElapsed.min(), 3)
    #     max_time = np.around(self.df.TimeElapsed.max(), 3)
    #     self.printit(f'Time range: {min_time}-{max_time} sec')

    # def ignore_slow_moving_detections(self, cutoff_speed: float):
    #     """Ignores slow moving vehicles"""
    #     if 'Speed' in self.df.columns:
    #         ibool = (self.df['Speed'] < cutoff_speed)
    #         self.df.drop(ibool[ibool].index, inplace=True)

    # # def ignore_short_lived_objects(self, cutoff_speed: float):
    # #     """Ignores slow moving vehicles"""
    # #     ibool = (self.df['Speed'] < cutoff_speed)
    # #     self.df.drop(ibool[ibool].index, inplace=True)

    # def replace_original_with_transformed(self):
    #     """replace original data with transformed"""
    #     orig_cnames = ['PositionX', 'PositionY', 'Heading', 'SpeedX', 'SpeedY']
    #     for cname in orig_cnames:
    #         if f'{cname}_Tr' in self.df.columns:
    #             self.df.rename(
    #                 mapper={cname: f'{cname}tmp'},
    #                 axis=1,
    #                 inplace=True
    #             )
    #             self.df.rename(
    #                 mapper={f'{cname}_Tr': cname},
    #                 axis=1,
    #                 inplace=True
    #             )
    #             self.df.drop(
    #                 columns=[f'{cname}tmp'],
    #                 inplace=True
    #             )

    # def remove_transformed(self):
    #     """remove transformed"""
    #     orig_cnames = ['PositionX', 'PositionY', 'Heading', 'SpeedX', 'SpeedY']
    #     ilist = [f'{k}_Tr' for k in orig_cnames]
    #     self.df.drop(columns=ilist, inplace=True, errors='ignore')

    # @property
    # def varnames(self):
    #     return list(self.df.columns)

    # def printit(self, istr: str, **kwargs):
    #     """Print command"""
    #     if self.verbose:
    #         print(self.name + ':', istr, flush=True, **kwargs)

    # def warnit(self, istr: str, **kwargs):
    #     """Print command"""
    #     warnings.warn(self.name + ':' + istr, **kwargs)

    # def draw_frame(
    #     self,
    #     ax,
    #     at_time_elapsed: float,
    #     time_padding: float
    # ) -> None:
    #     """Draws frame at this time"""
    #     # check time elapsed present
    #     assert 'TimeElapsed' in self.df.columns, 'Need TimeElapsed before draw_frame!'

    #     # get detected objets within the time frame
    #     time_low = at_time_elapsed - time_padding
    #     time_upp = at_time_elapsed + time_padding
    #     tbool = self.df.TimeElapsed.between(time_low, time_upp)
    #     xbool = self.df.PositionX.between(*ax.get_xlim())
    #     ybool = self.df.PositionY.between(*ax.get_ylim())
    #     dfshort = self.df[(tbool) & (xbool) & (ybool)].copy()
    #     dfshort.drop_duplicates(['ObjectId'], keep='last', inplace=True)

    #     # iter and draw all the detected objects
    #     for _, irow in dfshort.iterrows():
    #         centroid = (irow['PositionX'], irow['PositionY'])
    #         width = CAR_WIDTH if 'Width' not in dfshort.columns else irow.Width
    #         length = CAR_WIDTH if 'Length' not in dfshort.columns else irow.Length
    #         alpha = CAR_ALPHA if 'UncertaintyNorm' not in dfshort.columns else irow.UncertaintyNorm
    #         rot_transform = ax.transData
    #         lowerleft_loc = (
    #             centroid[0] - width / 2.,
    #             centroid[1] - length / 2.
    #         )
    #         angle = (irow['Heading'] + 90) % 360.
    #         rot_transform = mpl.transforms.Affine2D().rotate_deg_around(
    #             centroid[0],
    #             centroid[1],
    #             angle
    #         ) + ax.transData
    #         ibox = patches.FancyBboxPatch(
    #             xy=lowerleft_loc,
    #             width=width,
    #             height=length,
    #             boxstyle=BOX_STYLE,
    #             transform=rot_transform,
    #             ec='none',
    #             fc=self.clr,
    #             alpha=alpha
    #         )
    #         ax.add_patch(ibox)
    #         if self.show_ids:
    #             ax.text(
    #                 centroid[0],
    #                 centroid[1],
    #                 str(int(irow['ObjectId']))[-2:],
    #                 fontsize=3.5,
    #                 ha='center',
    #                 va='center',
    #                 color='w'
    #             )

    # for fpath, (fig, ax) in zip(fpaths, figures):
    #     leg = ax.legend(
    #         handles=list_of_patches,
    #         # handlelength=1.5,
    #         fontsize=9,
    #         loc=3,
    #         borderaxespad=0
    #     )
    #     fig.savefig(fpath, dpi=200, bbox_inches='tight')
    # return fpaths, figures
    # for fpath, (fig, _) in zip(fpaths, figures):
    #     fig.savefig(fpaths, dpi=160, bbox_inches='tight')

    # def plot_data(self, ax, projected: bool = True, *args, **kwargs) -> None:
    #     """plot the path on xy plane this run"""
    #     if projected:
    #         ax.plot(self.df.loc[:, 'PositionX'],
    #             self.df.loc[:, 'PositionY'],
    #             *args, **kwargs)
    #     else:
    #         ax.plot(self.df.loc[:, 'Longitude'],
    #                 self.df.loc[:, 'PositionY'],
    #                 *args, **kwargs)

#     def plot_time_history(
#             self,
#             vars_to_plot: List[str],
#             lstyle: str,
#             fig=None,
#             ax=None
#     ):
#         if fig is None:
#             ncols = len(vars_to_plot) // 2 + len(vars_to_plot) % 2
#             fig, ax = plt.subplots(
#                 ncols, 2,
#                 figsize=(9, 1.5 + ncols * 1),
#                 sharex=True
#             )
#             ax = ax.flatten()
#         assert 'TimeElapsed' in self.df.columns, 'Need TimeElapsed for time series'
#         for i, ivar in enumerate(vars_to_plot):
#             if ivar in self.df.columns:
#                 ax[i].plot(
#                     self.df['TimeElapsed'],
#                     self.df[ivar],
#                     lstyle,
#                     markersize=1.
#                 )
#                 ax[i].set_ylabel(ivar)
#                 ax[i].grid(True)
#         fig.tight_layout()
#         return fig, ax

#     def draw_namebox(self, ax, xydata: Tuple[float, float], **kwargs) -> None:
#         """Draw a box with name of the sensor"""
#         ax.text(
#             *xydata,
#             self.name,
#             transform=ax.transAxes,
#             fontsize=10,
#             **kwargs
#         )

#     def draw_frame(
#         self,
#         ax,
#         at_time: datetime.time,
#         time_padding: float,
#         show_id: bool = False,
#         use_alpha: bool = False,
#         **kwargs
#     ) -> None:
#         """Draws frame at this time"""
#         time_low = at_time - datetime.timedelta(seconds=time_padding)
#         time_upp = at_time + datetime.timedelta(seconds=time_padding)
#         dfshort = self.df.between_time(time_low.time(), time_upp.time())
#         # print(dfshort.shape[0])
#         for _, irow in dfshort.iterrows():
#             centroid = (irow['PositionX'], irow['PositionY'])
#             lowerleft_loc = (centroid[0], centroid[1])
#             width, length = (1.75, 1.75)
#             rot_transform = ax.transData
#             if self.object_shape == 'rectangle':
#                 width = irow[self.width_col]
#                 length = irow[self.length_col]
#                 lowerleft_loc = (
#                     centroid[0] - width / 2.,
#                     centroid[1] - length / 2.
#                 )
#                 angle = (irow[self.heading_col] + np.pi / 2) % (2.0 * np.pi)
#                 rot_transform = mpl.transforms.Affine2D().rotate_deg_around(
#                     centroid[0],
#                     centroid[1],
#                     np.degrees(angle)
#                 ) + ax.transData
#             alpha = 0.8
#             if self.quality_col in self.df.columns:
#                 if use_alpha:
#                     alpha = irow[self.quality_col]
#             ibox = patches.FancyBboxPatch(
#                 xy=lowerleft_loc,
#                 width=width,
#                 height=length,
#                 boxstyle=self.boxstyle,
#                 transform=rot_transform,
#                 alpha=alpha,
#                 **kwargs
#             )
#             ax.add_patch(ibox)
#             if self.id_col in self.df.columns:
#                 if show_id:
#                     ax.text(
#                         irow['PositionX'],
#                         irow['PositionY'],
#                         irow[self.id_col],
#                         fontsize=9,
#                         ha='center',
#                         va='center',
#                         color='w'
#                     )

# #     def get_object_list(self, states: List[str]) -> pd.DataFrame:
# #         """Returns object list"""
# #         for iname in states:
# #             out_str = f'{iname} not found! Choose among {self.states}'
# #             assert iname in self.states, out_str
# #         olist_data = self.df.loc[:, states].values.tolist()
# #         olist_index = self.df.index
# #         out_df = pd.DataFrame({'ObjectList': olist_data}, index=olist_index)
# #         return out_df

# #     def __str__(self) -> str:
# #         out_str = f'----{self.name}----\n'
# #         test_date = self.df.index.max().date().strftime('%F')
# #         out_str += f'Date of test : {test_date}\n'
# #         start_time = self.df.index.min().time().strftime('%H:%M:%S')
# #         end_time = self.df.index.max().time().strftime('%H:%M:%S')
# #         out_str += f'Time range   : {start_time}-{end_time}\n'
# #         out_str += f'# of entries : {self.df.shape[0]}\n'
# #         return out_str


# def merge_traffic_sensor_data(
#     list_of_sensors: List[TrafficSensor]
# ) -> pd.DataFrame:
#     """Merge data from two sensors"""
#     list_of_dfs = []
#     for ith_sensor in list_of_sensors:
#         ith_df = ith_sensor.get_object_list(ith_sensor.om.obs_names)
#         ith_df['SensorID'] = [ith_sensor.name] * ith_df.shape[0]
#         list_of_dfs.append(ith_df)
#     sdf = pd.concat(list_of_dfs)
#     sdf['SensorID'] = sdf['SensorID'].astype('category')
#     sdf.sort_index(inplace=True)
#     sdf['TimeElapsed'] = sdf.index.to_series().diff().dt.total_seconds()
#     sdf['TimeElapsed'] = sdf['TimeElapsed'].fillna(0.).cumsum()
#     return sdf
