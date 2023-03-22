#!/usr/bin/env python
"""Traffic Intersection class """

# pylint: disable=invalid-name
from typing import Tuple, Sequence, Optional
import warnings
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString, Point
from shapely.ops import split
from shapely.affinity import rotate
from shapely.errors import ShapelyDeprecationWarning
import shapely
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class TrafficIntersection:
    """Base class defining a traffic intersection"""

    def __init__(
        self,
        extent: Tuple[float, float, float, float]
    ) -> None:

        self.extent = extent  # (xmin, xmax, ymin, ymax)
        col_names = ('zone', 'type', 'orientation')
        self._df = pd.DataFrame({}, columns=col_names)
        self._df.index.names = ['name']
        self.center = ((self.extent[0] + self.extent[1]) / 2,
                       (self.extent[2] + self.extent[3]) / 2)
        self.width = (self.extent[1] - self.extent[0],
                      self.extent[3] - self.extent[2])
        corners = []
        for ix in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            izipper = zip(self.center, self.width, ix)
            corners.append([i + j * k for i, j, k in izipper])
        # left = [ix+iy*iz for ix,iy,iz in zip(self.center, self.width, [1,-1])]
        # left_bottom = (self.center[0] - self.width[0] / 2, self.center[1] - self.width[1] / 2)
        # right_bottom = (self.center[0] + self.width[0] / 2, self.center[1] - width[1] / 2)
        # right_upper = (center[0] + width[0] / 2, center[1] + width[1] / 2)
        # left_upper = (center[0] - width[0] / 2, center[1] + width[1] / 2)
        # [left_bottom, right_bottom, right_upper, left_upper]

        self.add_polygon_zone('world', corners)

    @property
    def xbound(self):
        """Returns bounds in x direction"""
        return [self.extent[0], self.extent[1]]

    @property
    def ybound(self):
        """Returns bounds in x direction"""
        return [self.extent[2], self.extent[3]]

    def add_polygon_zone(
        self,
        name: str,
        corners: Sequence[Tuple[float, float]],
        ztype: Optional[str] = None,
        orientation: float = np.nan
    ) -> None:
        """Creates a polygon zone within the intersection"""
        assert len(corners) > 2, 'Polygon needs atleast three points!'
        self._df.loc[name, 'zone'] = Polygon(corners)
        self._df.loc[name, 'type'] = ztype
        self._df.loc[name, 'orientation'] = orientation

    def add_circular_zone(
        self,
        name: str,
        center: Tuple[float, float],
        radius: Optional[float] = None,
        ztype: Optional[str] = None,
        orientation: float = np.nan,
    ) -> None:
        """Creates a polygon zone within the intersection"""
        assert len(center) == 2, 'Point/Circle zone needs 2d coordinates!'
        if radius is not None:
            self._df.loc[name, 'zone'] = Point(np.array(center)).buffer(radius)
        else:
            self._df.loc[name, 'zone'] = Point(np.array(center))
        self._df.loc[name, 'type'] = ztype
        self._df.loc[name, 'orientation'] = orientation

    def add_arc_zone(
        self,
        name: str,
        center: Tuple[float, float],
        radius: float = 1.,
        start_angle: float = 0.,
        end_angle: float = 45.,
        ztype: Optional[str] = None,
        orientation: float = np.nan,
        reverse: bool = False
    ) -> None:
        """Creates a arc zone within the intersection"""
        assert len(center) == 2, 'Point/Circle zone needs 2d coordinates!'
        point_a = (center[0] + radius + 10., center[1])
        x_axis = LineString([center, point_a])
        left_border = rotate(x_axis, start_angle, origin=center)
        right_border = rotate(x_axis, end_angle, origin=center)
        pt_a, pt_b = list(np.array(left_border.coords))
        pt_c, pt_d = list(np.array(right_border.coords))
        spliter = LineString([pt_a, pt_b, pt_c, pt_d])
        circle = Point(center).buffer(radius)
        if reverse:
            self._df.loc[name, 'zone'] = split(circle, spliter).geoms[0]
        else:
            self._df.loc[name, 'zone'] = split(circle, spliter).geoms[1]
        self._df.loc[name, 'type'] = ztype
        self._df.loc[name, 'orientation'] = orientation

    def describe(self, name: str) -> None:
        """Print basic details about this zone"""
        self.check_zone_name(name)
        zone = self.df.loc[name, 'zone']
        if zone.geom_type == 'Polygon':
            print(f"Centroid: {zone.centroid}")
            print(f"Area: {zone.area}")
            print(f"Bounding Box: {zone.bounds}")
            print(f"Exterior Length: {round(zone.exterior.length,4)}")

    def plot_this_zone(self, axs, iname: str, *args, **kwargs) -> None:
        """Plots this zones on the given axis"""
        self.check_zone_name(iname)
        izone = self._df.loc[iname, 'zone']
        if izone.geom_type == 'Point':
            print(izone.x, izone.y)
            axs.plot(izone.x, izone.y, *args, **kwargs)
        elif izone.geom_type == 'Polygon':
            x_coord, y_coord = izone.exterior.xy
            axs.fill(x_coord, y_coord, *args, **kwargs)
        axs.set_xlim([self.extent[0], self.extent[1]])
        axs.set_ylim([self.extent[2], self.extent[3]])

    def plot_this_type(self, axs, itype: str, *args, **kwargs) -> None:
        """Plots this type of zones on the given axis"""
        self.check_type_name(itype)
        for izone in self._df.loc[self._df['type'] == itype, 'zone']:
            if izone.geom_type == 'Point':
                axs.plot(izone.x, izone.y, *args, **kwargs)
            elif izone.geom_type == 'Polygon':
                x_coord, y_coord = izone.exterior.xy
                axs.fill(x_coord, y_coord, *args, **kwargs)
        axs.set_xlim([self.extent[0], self.extent[1]])
        axs.set_ylim([self.extent[2], self.extent[3]])

    def within_this_zone(self, zone_name, xlocs, ylocs) -> None:
        """Check if point is within this zone"""
        self.check_zone_name(zone_name)
        vec_func = np.vectorize(self.df.loc[zone_name, 'zone'].contains)
        points = np.empty(len(xlocs), dtype='object')
        for i, (ix, iy) in enumerate(zip(xlocs, ylocs)):
            points[i] = Point(ix, iy)
        return vec_func(points)

    def within_this_type(self, type_name, xlocs, ylocs) -> None:
        """Check if point is within this zone"""
        self.check_type_name(type_name)
        out_bool = False
        for izone in self.df.loc[self.df['type'] == type_name].index.tolist():
            out_bool = out_bool | self.within_this_zone(izone, xlocs, ylocs)
        return out_bool

    def check_zone_name(self, name):
        """Check if name exists in the dataframe"""
        valid_names = self._df.index.to_list()
        err_str = f'{name} not found!\nChoose among {valid_names}'
        assert name in valid_names, err_str

    def check_type_name(self, name):
        """Check if name exists in the dataframe"""
        valid_names = self._df['type'].unique()
        err_str = f'{name} not found!\nChoose among {valid_names}'
        assert name in valid_names, err_str

    @property
    def df(self):
        """Getter for df"""
        return self._df


def get_csprings_parking_lot(big: bool = True):
    """Colorado springs parking lot"""
    silver_pole_coords = (0, 0)
    silver_pole_hor_angle = 60.  # from north
    black_pole_coords = (-2, 2)
    black_pole_hor_angle = 85  # from north
    radar_fov = 110  # degrees
    radar_range = 150.
    if big:
        cs = TrafficIntersection(extent=[-75, 85, -10, 150])
    else:
        cs = TrafficIntersection(extent=[-70, 70, -10, 130])
    cs.add_circular_zone('radar_silver', silver_pole_coords, ztype='radar')
    cs.add_circular_zone('radar_black', black_pole_coords, ztype='radar')
    cs.add_arc_zone('radar_silver_coverage', silver_pole_coords,
                    radius=radar_range,
                    start_angle=silver_pole_hor_angle - radar_fov / 2,
                    end_angle=silver_pole_hor_angle + radar_fov / 2,
                    ztype='radar_coverage', reverse=False)
    cs.add_arc_zone('radar_black_coverage', black_pole_coords,
                    radius=radar_range,
                    start_angle=black_pole_hor_angle - radar_fov / 2,
                    end_angle=black_pole_hor_angle + radar_fov / 2,
                    ztype='radar_coverage', reverse=False)
    cs.add_polygon_zone('train_tracks',
                        [(70, 25), (0, 135), (20, 155), (90, 45)],
                        ztype='exclusions')
    return cs
