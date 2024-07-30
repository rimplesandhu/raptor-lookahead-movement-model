#!/usr/bin/env python
"""Traffic sensor class"""

# pylint: disable=invalid-name
import glob
import os
import warnings
from functools import partial
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import cartopy.crs as ccrs

from .utils import run_loop
warnings.simplefilter('ignore')

GEO_CRS = ccrs.CRS('EPSG:4326')
PROJ_CRS = ccrs.UTM(13)  # 'EPSG:3857'
CAR_WIDTH = 2.5
CAR_LENGTH = 4.5
BOX_STYLE = patches.BoxStyle('round', rounding_size=1.)


class TrafficSensor:
    """Base class for defining a traffic sensor that produces an object list"""

    def __init__(
        self,
        name: str,
        tdata: list[datetime],
        xdata: list[float],
        ydata: list[float],
        speed: list[float],
        speed_x: list[float] | None = None,
        speed_y: list[float] | None = None,
        heading_deg: list[float] | None = None,
        object_id: list[int] | None = None,
        width: list[float] | None = None,
        length: list[float] | None = None,
        loc_lonlat: Tuple[float, float] | None = None,
        clr: str = 'b'
    ):

        # check compatibility of inputs
        assert len(tdata) == len(xdata), 'tdata/xdata size mismatch!'
        assert len(xdata) == len(ydata), 'xdata/ydata size mismatch!'
        assert len(ydata) == len(speed), 'ydata/speed size mismatch!'

        # basic info
        self.name = name
        self.clr = clr

        # some checks
        self.printit('Initiating sensor dataframe..')
        self.df = pd.DataFrame(dict(
            Time=pd.to_datetime(tdata),
            PositionX=xdata,
            PositionY=ydata,
            Speed=speed,
            Distance=np.sqrt(xdata**2+ydata**2)
        ))
        # self.df['Time'] = self.df['Time'].dt.round('1ms')

        # sensor location, compute lon/lat positions
        self.loc_sensor_lonlat = loc_lonlat
        self.compute_lonlats()
        self.loc_sensor_xy = (0., 0.)

        # deal with object id data
        if object_id is None:
            self.printit('No object_id supplied -> ObjectId=0 for all')
            self.df['ObjectId'] = 0
        else:
            self.df['ObjectId'] = object_id
        codes, _ = pd.factorize(self.df['ObjectId'])
        self.df['ObjectId'] = codes
        self.df['ObjectId'] = self.df['ObjectId'].astype(np.int32)
        self.df['ObjectId'] -= self.df['ObjectId'].min()
        self.printit(f'Contain {self.df.ObjectId.nunique()} object(s)')

        # deal with absense of heading or speedx/speedy data
        if heading_deg is not None:
            self.printit('Heading (deg) is assumed +ve CW from x axis!')
            # assert np.amax(heading_deg) > 180., 'Heading outside [180,-180)!'
            # assert np.amin(heading_deg) < -180., 'Heading outside [180,-180)!'
            self.df['Heading'] = heading_deg
            self.df.loc[self.df['Heading'] > 180., 'Heading'] -= 360
            self.df.loc[self.df['Heading'] <= -180., 'Heading'] += 360
        if speed_x is not None:
            assert speed_y is not None, 'Need both speedx and speedy!'
            self.df['SpeedX'] = speed_x
            self.df['SpeedY'] = speed_y
            if heading_deg is None:
                self.printit('Computing Heading using SpeedX/SpeedY..')
                self.df['Heading'] = np.degrees(
                    np.arctan2(speed_y, speed_x))
        else:
            assert speed_y is None, 'Need both speedx and speedy!'
            if heading_deg is not None:
                self.printit('Computing SpeedX/SpeedY using Speed/Heading..')
                self.df['SpeedX'] = speed * \
                    np.cos(np.radians(self.df['Heading']))
                self.df['SpeedY'] = speed * \
                    np.sin(np.radians(self.df['Heading']))

        # deal with width and length data
        if width is None:
            self.printit(f'No object width data supplied!')
        else:
            assert len(width) == len(xdata), 'width/xdata length mismatch!'
            self.df['Width'] = width
        if length is None:
            self.printit(f'No object length data supplied!')
        else:
            self.df['Length'] = length

        # data types for the dataframes
        self.df.set_index('Time', inplace=True, drop=True)
        self.df.sort_index(inplace=True)
        self.df.reset_index(drop=False, inplace=True)
        float64_cols = list(self.df.select_dtypes(include='float64'))
        self.df[float64_cols] = self.df[float64_cols].astype('float32')
        int64_cols = list(self.df.select_dtypes(include='int64'))
        self.df[int64_cols] = self.df[int64_cols].astype('int32')
        self.dfraw = self.df.copy()
        self.print_time_summary()

        # initiation done
        self.printit(f'Contains {self.df.shape[0]} detections. All done.')

        # if time_minmax is not None:
        #     self.df = self.df.between_time(*time_minmax)
        # self.states = list(self.df.columns)
        # assert 'PositionX' in self.states, f'Need {'PositionX'} in data_df!'
        # assert 'PositionY' in self.states, f'Need {'PositionY'} in data_df!'
        # self.df.set_index(pd.to_datetime(time_series), inplace=True)

        # if heading_deg is not None:
        #     self.shape = 'rectangle'
        #
        # else:
        #     self.shape = 'circle'
        #     self.boxstyle = patches.BoxStyle('circle', pad=0.3)

    def compute_lonlats(self) -> None:
        """Compute latitute and longitude of sensor data"""
        if self.loc_sensor_lonlat is not None:
            assert len(
                self.loc_sensor_lonlat) == 2, 'Need 2 floats in loc_lonlat!'
            # get position in meters in proj ref system
            loc_xy = PROJ_CRS.transform_points(
                src_crs=GEO_CRS,
                x=np.array([self.loc_sensor_lonlat[0]]),
                y=np.array([self.loc_sensor_lonlat[1]])
            )[0]
            pos_x = self.df['PositionX'] + loc_xy[0]
            pos_y = self.df['PositionY'] + loc_xy[1]

            # compute lon and lats
            lonlats = GEO_CRS.transform_points(
                src_crs=PROJ_CRS,
                x=pos_x,
                y=pos_y
            )
            self.df['Longitude'] = lonlats[:, 0]
            self.df['Latitude'] = lonlats[:, 1]

    def print_time_summary(self) -> None:
        """Print time summary"""
        self.printit(f'Starts at {self.df['Time'].min()}')
        self.printit(f'  Ends at {self.df['Time'].max()}')

    def add_time_bias(self, time_sec: float):
        """Add seconds to time"""
        self.printit(f'Adding {time_sec} sec to time..')
        self.df['Time'] = self.dfraw['Time'] + timedelta(seconds=time_sec)
        self.printit('Make sure to recompute time elapsed!')

        # if 'TimeElapsed' in self.df.columns:
        #     self.df['TimeElapsed'] += time_sec

    def transform(
            self,
            angle_deg: float,
            xdist: float = 0.,
            ydist: float = 0.
    ) -> None:
        """Rotate the data by some angle"""
        # heading
        self.df['Heading'] = self.dfraw['Heading'] + angle_deg
        self.df.loc[self.df['Heading'] > 180., 'Heading'] -= 360.
        self.df.loc[self.df['Heading'] <= -180., 'Heading'] += 180.

        # position
        xnew, ynew = self._rotate_clockwise(
            self.dfraw.PositionX,
            self.dfraw.PositionY,
            angle_deg=angle_deg
        )
        self.df['PositionX'] = xnew
        self.df['PositionY'] = ynew
        self.compute_lonlats()

        # speed
        xnew, ynew = self._rotate_clockwise(
            xvals=self.dfraw.SpeedX,
            yvals=self.dfraw.SpeedY,
            angle_deg=angle_deg
        )
        self.df['SpeedX'] = xnew
        self.df['SpeedY'] = ynew

        # translate
        self.df['PositionX'] += xdist
        self.df['PositionY'] += ydist
        self.loc_sensor_xy = (xdist, ydist)
        self.compute_lonlats()

    def _rotate_clockwise(self, xvals, yvals, angle_deg: float):
        """rotarte clockwise"""
        angle_rad = np.radians(-angle_deg)
        xnew = xvals * np.cos(angle_rad)
        xnew -= yvals * np.sin(angle_rad)
        ynew = xvals * np.sin(angle_rad)
        ynew += yvals * np.cos(angle_rad)
        return xnew, ynew

    def compute_timeelapsed(self, base_time: datetime, round_to: str = '100ms') -> None:
        """add time elapsed and time diff columns"""
        self.df['Time'] = self.df['Time'].dt.round(round_to)
        self.df['TimeElapsed'] = self.df['Time'] - base_time
        self.df['TimeElapsed'] = self.df['TimeElapsed'].dt.total_seconds()
        self.df['TimeElapsed'] = self.df['TimeElapsed'].round(decimals=3)
        # self.df['TimeDiff'] = self.df['TimeElapsed'].diff().fillna(0.)
        self.printit(f'Starts at {self.df.TimeElapsed.min()} sec')
        self.printit(f'  Ends at {self.df.TimeElapsed.max()} sec')

    def ignore_slow_moving_detections(self, cutoff_speed: float):
        """Ignores slow moving vehicles"""
        ibool = (self.df['Speed'] < cutoff_speed)
        self.df.drop(ibool[ibool].index, inplace=True)

    # def ignore_short_lived_objects(self, cutoff_speed: float):
    #     """Ignores slow moving vehicles"""
    #     ibool = (self.df['Speed'] < cutoff_speed)
    #     self.df.drop(ibool[ibool].index, inplace=True)

    def get_object_list(
            self,
            varnames: List[str],
            timeelapsed_range=None
    ) -> pd.DataFrame:
        """Returns object list"""
        for iname in varnames:
            out_str = f'{iname} not found! Choose among {self.varnames}'
            assert iname in self.varnames, out_str
        if timeelapsed_range is not None:
            assert 'TimeElapsed' in self.varnames, 'run compute_timeelapsed() first!'
            ibool = self.df['TimeElapsed'].between(*timeelapsed_range)
        else:
            ibool = self.df.index

        out_df = pd.DataFrame({
            'Time': self.df.loc[ibool, 'Time'].values,
            'TimeElapsed': self.df.loc[ibool, 'TimeElapsed'].values,
            'Data': self.df.loc[ibool, varnames].values.tolist()
        })
        out_df['Sensor'] = pd.Series(
            [self.name]*out_df.shape[0], dtype='category')
        return out_df

    @property
    def varnames(self):
        return list(self.df.columns)

    def reinitialize_data(self):
        """Goes back to raw data"""
        self.df = self.dfraw.copy()

    def printit(self, istr: str, **kwargs):
        """Print command"""
        print(self.name + ':', istr, flush=True, **kwargs)

    def warnit(self, istr: str, **kwargs):
        """Print command"""
        warnings.warn(self.name + ':' + istr, **kwargs)

    def draw_frame(
        self,
        ax,
        at_time_elapsed: float,
        time_padding: float,
        **kwargs
    ) -> None:
        """Draws frame at this time"""
        # check time elapsed present
        assert 'TimeElapsed' in self.df.columns, 'Need TimeElapsed before draw_frame!'

        # get detected objets within the time frame
        time_low = at_time_elapsed - time_padding
        time_upp = at_time_elapsed + time_padding
        tbool = self.df.TimeElapsed.between(time_low, time_upp)
        xbool = self.df.PositionX.between(*ax.get_xlim())
        ybool = self.df.PositionY.between(*ax.get_ylim())
        dfshort = self.df[(tbool) & (xbool) & (ybool)].copy()
        dfshort.drop_duplicates(['ObjectId'], keep='last', inplace=True)

        # iter and draw all the detected objects
        for _, irow in dfshort.iterrows():
            centroid = (irow['PositionX'], irow['PositionY'])
            width = CAR_WIDTH if 'Width' not in dfshort.columns else irow.Width
            length = CAR_WIDTH if 'Length' not in dfshort.columns else irow.Length
            rot_transform = ax.transData
            lowerleft_loc = (
                centroid[0] - width / 2.,
                centroid[1] - length / 2.
            )
            angle = (irow['Heading'] + 90) % 360.
            rot_transform = mpl.transforms.Affine2D().rotate_deg_around(
                centroid[0],
                centroid[1],
                angle
            ) + ax.transData
            ibox = patches.FancyBboxPatch(
                xy=lowerleft_loc,
                width=width,
                height=length,
                boxstyle=BOX_STYLE,
                transform=rot_transform,
                ec='none',
                fc=self.clr,
                **kwargs
            )
            ax.add_patch(ibox)
            ax.text(
                centroid[0],
                centroid[1],
                str(int(irow.ObjectId))[-2:],
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
    **kwargs
):
    """Create animation from a list of dataframes"""
    # initiate
    istr = f'{round(list_of_etimes[0], 3)}-{round(list_of_etimes[-1], 3)}'
    print(f'Creating frames within time {istr}..')

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

    # run the loop
    for isensor in list_of_sensors:
        def draw_frame_fn(ituple):
            return isensor.draw_frame(
                ax=ituple[0],
                at_time_elapsed=ituple[1],
                time_padding=(list_of_etimes[1] - list_of_etimes[0])/2.,
                **kwargs
            )
        _ = run_loop(
            draw_frame_fn,
            loop_tuple,
            desc=isensor.name,
            ncores=ncores,
            disable=False
        )

    # create legend manually
    list_of_patches = []
    for isensor in list_of_sensors:
        ipatch = patches.Patch(
            color=isensor.clr,
            label=isensor.name
        )
        list_of_patches.append(ipatch)

    # finish and save

    def _save_frames(ix):
        fpath, (fig, ax) = ix
        _ = ax.legend(
            handles=list_of_patches,
            # handlelength=1.5,
            fontsize=9,
            loc=3,
            borderaxespad=0
        )
        fig.savefig(fpath, dpi=dpi, bbox_inches='tight')

    _ = run_loop(_save_frames, list(zip(fpaths, figures)),
                 ncores=ncores, desc='Save')
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
