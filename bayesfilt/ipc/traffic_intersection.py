#!/usr/bin/env python
"""Traffic Intersection class """

# pylint: disable=invalid-name
from typing import Tuple, Sequence, Optional
import warnings
import numpy as np
import pandas as pd
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString, Point
from shapely.ops import split, unary_union
from shapely.affinity import rotate
from shapely.errors import ShapelyDeprecationWarning
import shapely
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


class TrafficIntersection:
    """Base class defining a traffic intersection"""

    def __init__(
        self,
        center_lonlat: Tuple[float, float],
        extent_meters: Tuple[float, float, float, float]
    ) -> None:

        self.extent = extent_meters  # (xmin, xmax, ymin, ymax)
        self.center_lonlat = center_lonlat
        self.object_cname = 'Object'
        self.type_name = 'Type'
        self.heading_name = 'Heading'
        col_names = (self.object_cname, self.type_name, self.heading_name)
        self._df = pd.DataFrame({}, columns=col_names)
        self._df.index.names = ['Name']
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

        self.add_polygon('world', corners)

    @property
    def xbound(self):
        """Returns bounds in x direction"""
        return [self.extent[0], self.extent[1]]

    @property
    def ybound(self):
        """Returns bounds in x direction"""
        return [self.extent[2], self.extent[3]]

    def add_lane(
        self,
        name: str,
        centerline_point1: Tuple[float, float],
        centerline_point2: Tuple[float, float],
        lanewidth: float
    ) -> None:
        """Creates a polygon zone within the intersection"""
        x1, y1 = centerline_point1
        x2, y2 = centerline_point2
        w = lanewidth/2.
        theta = np.arctan2(y2-y1, x2-x1)
        p1 = (x1+w*np.sin(theta), y1-w*np.cos(theta))
        p2 = (x2+w*np.sin(theta), y2-w*np.cos(theta))
        p3 = (x2-w*np.sin(theta), y2+w*np.cos(theta))
        p4 = (x1-w*np.sin(theta), y1+w*np.cos(theta))
        self._df.loc[name, self.object_cname] = Polygon([p1, p2, p3, p4])
        self._df.loc[name, self.type_name] = 'lane'
        self._df.loc[name, self.heading_name] = np.degrees(theta)

    def add_polygon(
        self,
        name: str,
        corners: Sequence[Tuple[float, float]],
        zone_type: Optional[str] = None,
        orientation: float = np.nan
    ) -> None:
        """Creates a polygon zone within the intersection"""
        assert len(corners) > 2, 'Polygon needs atleast three points!'
        self._df.loc[name, self.object_cname] = Polygon(corners)
        self._df.loc[name, self.type_name] = zone_type
        self._df.loc[name, self.heading_name] = orientation

    def add_circle(
        self,
        name: str,
        center: Tuple[float, float],
        radius: Optional[float] = None,
        zone_type: Optional[str] = None,
        orientation: float = np.nan,
    ) -> None:
        """Creates a polygon zone within the intersection"""
        assert len(center) == 2, 'Point/Circle zone needs 2d coordinates!'
        if radius is not None:
            self._df.loc[name, self.object_cname] = Point(
                np.array(center)).buffer(radius)
        else:
            self._df.loc[name, self.object_cname] = Point(np.array(center))
        self._df.loc[name, self.type_name] = zone_type
        self._df.loc[name, self.heading_name] = orientation

    def add_arc(
        self,
        name: str,
        center: Tuple[float, float],
        radius: float = 1.,
        start_angle: float = 0.,
        end_angle: float = 45.,
        zone_type: Optional[str] = None,
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
            self._df.loc[name, self.object_cname] = split(
                circle, spliter).geoms[0]
        else:
            self._df.loc[name, self.object_cname] = split(
                circle, spliter).geoms[1]
        self._df.loc[name, self.type_name] = zone_type
        self._df.loc[name, self.heading_name] = orientation

    def describe(self, name: str) -> None:
        """Print basic details about this zone"""
        self.check_zone_name(name)
        zone = self.df.loc[name, self.object_cname]
        if zone.geom_type == 'Polygon':
            print(f"Centroid: {zone.centroid}")
            print(f"Area: {zone.area}")
            print(f"Bounding Box: {zone.bounds}")
            print(f"Exterior Length: {round(zone.exterior.length, 4)}")

    def plot_this_zone(self, axs, iname: str, *args, **kwargs) -> None:
        """Plots this zones on the given axis"""
        self.check_zone(iname)
        izone = self.df.loc[iname, self.object_cname]
        if izone.geom_type == 'Point':
            print(izone.x, izone.y)
            axs.plot(izone.x, izone.y, *args, **kwargs)
        elif izone.geom_type == 'Polygon':
            x_coord, y_coord = izone.exterior.xy
            axs.fill(x_coord, y_coord, *args, **kwargs)
        # axs.set_xlim([self.extent[0], self.extent[1]])
        # axs.set_ylim([self.extent[2], self.extent[3]])

    def plot_this_type(self, axs, itype: str, *args, **kwargs) -> None:
        """Plots this type of zones on the given axis"""
        self.check_zonetype(itype)
        for izone in self._df.loc[self.df[self.type_name] == itype, self.object_cname]:
            if izone.geom_type == 'Point':
                axs.plot(izone.x, izone.y, *args, **kwargs)
            elif izone.geom_type == 'Polygon':
                x_coord, y_coord = izone.exterior.xy
                axs.fill(x_coord, y_coord, *args, **kwargs)
        # axs.set_xlim([self.extent[0], self.extent[1]])
        # axs.set_ylim([self.extent[2], self.extent[3]])

    def within_this_zone(self, zone, xlocs, ylocs) -> None:
        """Check if point is within this zone"""
        list_of_points = [Point(ix, iy) for ix, iy in zip(xlocs, ylocs)]
        this_zone = self.get_zone(zone=zone)
        bool_list = [this_zone.contains(ix) for ix in list_of_points]
        return np.array(bool_list)

    def within_this_zonetype(self, zone, xlocs, ylocs) -> None:
        """Check if point is within this zone"""
        list_of_points = [Point(ix, iy) for ix, iy in zip(xlocs, ylocs)]
        this_zone = self.get_zonetype(zone=zone)
        bool_list = [this_zone.contains(ix) for ix in list_of_points]
        return np.array(bool_list)

    # def within_this_zone(self, zone, xlocs, ylocs) -> None:
    #     """Check if point is within this zone"""
    #     self.check_zone(zone)
    #     out_bool = False
    #     for izone in self.df.loc[self.df[self.type_name] == zone].index.tolist():
    #         out_bool = out_bool | self.within_this_zone(izone, xlocs, ylocs)
    #     return out_bool

    def get_zonetype(self, zone: str):
        """Returns shapley object of this type of zone"""
        self.check_zonetype(zone)
        zlist = self.df.loc[self.df[self.type_name] == zone, self.object_cname]
        return unary_union(zlist.tolist())

    def check_zonetype(self, name):
        """Check if name exists in the dataframe"""
        valid_names = self.df[self.type_name].unique()
        err_str = f'{name} not found!\nChoose among {valid_names}'
        assert name in valid_names, err_str

    def check_zone(self, name):
        """Check if name exists in the dataframe"""
        valid_names = self.df.index.to_list()
        err_str = f'{name} not found!\nChoose among {valid_names}'
        assert name in valid_names, err_str

    def get_zone(self, zone: str):
        """Returns shapley object of this type of zone"""
        self.check_zone(zone)
        zlist = self.df.loc[self.df.index == zone, self.object_cname]
        return unary_union(zlist.tolist())

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
    cs.add_circular_zone('radar_silver', silver_pole_coords, zone_type='radar')
    cs.add_circular_zone('radar_black', black_pole_coords, zone_type='radar')
    cs.add_arc_zone('radar_silver_coverage', silver_pole_coords,
                    radius=radar_range,
                    start_angle=silver_pole_hor_angle - radar_fov / 2,
                    end_angle=silver_pole_hor_angle + radar_fov / 2,
                    zone_type='radar_coverage', reverse=False)
    cs.add_arc_zone('radar_black_coverage', black_pole_coords,
                    radius=radar_range,
                    start_angle=black_pole_hor_angle - radar_fov / 2,
                    end_angle=black_pole_hor_angle + radar_fov / 2,
                    zone_type='radar_coverage', reverse=False)
    cs.add_polygon_zone('train_tracks',
                        [(70, 25), (0, 135), (20, 155), (90, 45)],
                        zone_type='exclusions')
    return cs
