""" Class defining the 3DEP data"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation


import sys
import os
from typing import Callable
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rio
import rasterio
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy import ndarray
import cartopy.crs as ccrs
from ssrs import Terrain
from ._base_data import BaseData


class Data3DEP(BaseData):
    """Class for downloading 3DEP data"""

    def __init__(
        self,
        lonlat_bounds: tuple[float, float, float, float],
        resolution: float,
        **kwargs
    ):
        BaseData.__init__(self, **kwargs)
        self.lonlat_bound = lonlat_bounds
        self.resolution = resolution
        self.dem = 'GroundElevation'
        self.terrain = Terrain(
            self.lonlat_bound,
            self.out_dir,
            print_verbose=False
        )

    def download(self) -> None:
        """Download 3DEP data"""
        self.terrain.download('DEM')

    @property
    def ds_geo(self):
        """Get xarray dataset containing the 3dep layers in geo crs"""
        dsg = rio.open_rasterio(self.terrain.get_raster_fpath('DEM'))
        dsg = dsg.to_dataset(name=self.dem).squeeze()
        return dsg

    @property
    def ds(self):
        """Returns xarray in proj crs"""
        ds = self.ds_geo.rio.reproject(
            self.proj_crs,
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

    @classmethod
    def download_function(cls, ix):
        with open(os.devnull, 'w', encoding='UTF-8') as f:
            lonlat_bnd, domain_dir, proj_crs, resolution = ix
            #print(lonlat_bnd, domain_dir, proj_crs, resolution)
            #sys.stdout = f
            dep3 = Data3DEP(
                lonlat_bounds=lonlat_bnd,
                resolution=resolution,
                proj_crs=proj_crs,
                out_dir=domain_dir
            )
            dep3.download()
            #sys.stdout = sys.__stdout__
        return dep3

    @classmethod
    def annotate_function(cls, ituple):
        xlocs, ylocs, _, dep3_obj = ituple
        xlocs_xr = xr.DataArray(xlocs, dims=['points'])
        ylocs_xr = xr.DataArray(ylocs, dims=['points'])
        return dep3_obj.ds.interp(
            x=xlocs_xr,
            y=ylocs_xr,
            method='linear',
            kwargs={'fill_value': None},
        )


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
