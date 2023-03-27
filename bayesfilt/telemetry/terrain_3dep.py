""" Base class for 3DEP data annotation """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
from typing import Callable
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import rioxarray as rio
import rasterio
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy import ndarray
import cartopy.crs as ccrs
from ssrs.terrain import Terrain


@dataclass
class Terrain3DEP:
    """Class for downloading 3DEP data"""
    lonlat_bound: tuple[float, float, float, float]
    out_dir: str
    proj_crs: str = 'ESRI:102008'
    resolution: float = 10
    terrain: Terrain | None = field(default=None, repr=False)
    layers: list[str] | None = field(default=None, repr=False)
    dem: str = field(default='GroundElevation', repr=False)

    def __post_init__(self):
        """Post initialization function"""
        self.out_dir = Path(self.out_dir)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.layers = 'DEM' if self.layers is None else self.layers
        self.terrain = Terrain(
            self.lonlat_bound,
            self.out_dir,
            print_verbose=False
        )

    def download(self):
        """Download 3DEP data"""
        self.terrain.download(self.layers)

    @property
    def ds_geo(self):
        """Get xarray dataset containing the 3dep layers in geo crs"""
        dsg = rio.open_rasterio(self.terrain.get_raster_fpath('DEM'))
        dsg = dsg.to_dataset(name=self.dem).squeeze()
        return dsg

    @property
    def ds_proj(self):
        """Returns xarray in proj crs"""
        ds = self.ds_geo.rio.reproject(
            rio.crs.crs_from_user_input(self.proj_crs),
            resolution=self.resolution,
            nodata=np.nan,
            Resampling=rasterio.enums.Resampling.bilinear
        )
        slope = compute_slope_degrees(ds[self.dem].values, self.resolution)
        aspect = compute_aspect_degrees(ds[self.dem].values, self.resolution)
        ds['GroundSlope'] = (('y', 'x'), slope)
        ds['GroundAspect'] = (('y', 'x'), aspect)
        ds['GroundSlope'].values[ds[self.dem].isnull()] = np.nan
        ds['GroundAspect'].values[ds[self.dem].isnull()] = np.nan
        ds = ds.drop_vars(['band', 'spatial_ref'], errors='ignore')
        return ds

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.__class__.__name__}: {outstr}')

    def printit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        print(f'{self.__class__.__name__}: {outstr}', flush=True)


def compute_slope_degrees(z_mat: np.ndarray, res: float):
    """ Calculate local terrain slope using 3x3 stencil
    Parameters:
    ----------
    z_mat : numpy array
        Contains elevation data in meters
    res: float
        Resolution in meters, assumed to be same in both directions
    Returns:
    --------
    numpy array containing slope in degrees
    """

    slope = np.empty_like(z_mat)
    slope[:, :] = np.nan
    z_1 = z_mat[:-2, 2:]  # upper left
    z_2 = z_mat[1:-1, 2:]  # upper middle
    z_3 = z_mat[2:, 2:]  # upper right
    z_4 = z_mat[:-2, 1:-1]  # center left
   # z5 = z[ 1:-1, 1:-1] # center
    z_6 = z_mat[2:, 1:-1]  # center right
    z_7 = z_mat[:-2, :-2]  # lower left
    z_8 = z_mat[1:-1, :-2]  # lower middle
    z_9 = z_mat[2:, :-2]  # lower right
    dz_dx = ((z_3 + 2 * z_6 + z_9) - (z_1 + 2 * z_4 + z_7)) / (8 * res)
    dz_dy = ((z_1 + 2 * z_2 + z_3) - (z_7 + 2 * z_8 + z_9)) / (8 * res)
    rise_run = np.sqrt(dz_dx**2 + dz_dy**2)
    slope[1:-1, 1:-1] = np.degrees(np.arctan(rise_run))
    return np.nan_to_num(slope)


def compute_aspect_degrees(z_mat: np.ndarray, res: float):
    """ Calculate local terrain aspect using 3x3 stencil
    Parameters:
    ----------
    z : numpy array
        Contains elevation data in meters
    res: float
        Resolution in meters, assumed to be same in both directions
    Returns:
    --------
    numpy array containing aspect in degrees
    """

    aspect = np.empty_like(z_mat)
    aspect[:, :] = np.nan
    z_1 = z_mat[:-2, 2:]  # upper left
    z_2 = z_mat[1:-1, 2:]  # upper middle
    z_3 = z_mat[2:, 2:]  # upper right
    z_4 = z_mat[:-2, 1:-1]  # center left
   # z5 = z[ 1:-1, 1:-1] # center
    z_6 = z_mat[2:, 1:-1]  # center right
    z_7 = z_mat[:-2, :-2]  # lower left
    z_8 = z_mat[1:-1, :-2]  # lower middle
    z_9 = z_mat[2:, :-2]  # lower right
    dz_dx = ((z_3 + 2 * z_6 + z_9) - (z_1 + 2 * z_4 + z_7)) / (8 * res)
    dz_dy = ((z_1 + 2 * z_2 + z_3) - (z_7 + 2 * z_8 + z_9)) / (8 * res)
    dz_dx[dz_dx == 0.] = 1e-10
    angle = np.degrees(np.arctan(np.divide(dz_dy, dz_dx)))
    angle_mod = 90. * np.divide(dz_dx, np.absolute(dz_dx))
    aspect[1:-1, 1:-1] = 180. - angle + angle_mod
    return np.nan_to_num(aspect)

# dem_name = DICT_OF_3DEP_VARS[DEM_3DEP_LAYER]
# ds_3dep = rio.open_rasterio(filepath)
# ds_3dep = ds_3dep.to_dataset(name=dem_name).squeeze()
# ds = ds_3dep.rio.reproject(
#     rio.crs.crs_from_user_input(proj_crs),
#     resolution=resolution,
#     nodata=np.nan,
#     Resampling=rasterio.enums.Resampling.bilinear
# )
# ds[dem_name].attrs = {}
# slope = calcSlopeDegrees(ds[dem_name].values, res=resolution)
# aspect = calcAspectDegrees(ds[dem_name].values, res=resolution)
# ds['GroundSlope'] = (('y', 'x'), slope)
# ds['GroundAspect'] = (('y', 'x'), aspect)
# ds['GroundSlope'].values[ds[dem_name].isnull()] = np.nan
# ds['GroundAspect'].values[ds[dem_name].isnull()] = np.nan
# ds = ds.drop_vars(['band', 'spatial_ref'], errors='ignore')
