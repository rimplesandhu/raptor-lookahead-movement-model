""" Base class for 3DEP data annotation """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
import os
import sys
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
from numpy import ndarray, datetime64
import cartopy.crs as ccrs
from ssrs import HRRR
from ._base_data import BaseGeoData


class DataHRRR(BaseGeoData):
    """Class for downloading HRRR data"""
    hrrr_proj4 = '+ellps=WGS84 +a=6371229.0 +b=6371229.0 +proj=lcc +lon_0=262.5 +lat_0=38.5 +x_0=0.0 +y_0=0.0 +lat_1=38.5 \
                +lat_2=38.5 +no_defs'
    hrrr_crs = hrrr_proj4
    variables = {
        ':UGRD:10 m': 'WindSpeedU_10m',
        ':VGRD:10 m': 'WindSpeedV_10m',
        ':UGRD:80 m': 'WindSpeedU_80m',
        ':VGRD:80 m': 'WindSpeedV_80m',
        ':TMP:surface': 'Temperature_0m',
    }

    def __init__(
        self,
        time_utc: datetime64,
        **kwargs
    ):
        BaseGeoData.__init__(self, proj_crs=self.hrrr_crs, **kwargs,)
        #self.proj_crs = self.hrrr_crs
        self.time_utc = np.datetime64(time_utc)
        self.time_str = np.datetime_as_string(self.time_utc, unit='h',
                                              timezone='UTC')
        self.filepath = self.out_dir / f'{self.time_str}.nc'

    @classmethod
    def refine_hrrr_xarray(cls, ids):
        """ Function to refine hrrr xarray loaded from grib files"""
        ids = ids.set_coords('gribfile_projection')
        drop_these_coords = ['step', 'valid_time',
                             'heightAboveGround', 'surface']
        ids = ids.drop_vars(drop_these_coords, errors='ignore')
        ids.time.attrs = {}
        ids.latitude.attrs = {}
        ids.attrs = {k: v for k, v in ids.attrs.items()
                     if 'grib' not in k.lower()}
        for ix in list(ids.keys()):
            ids[ix].attrs = {k: v for k, v in ids[ix].attrs.items()
                             if 'grib' not in k.lower()}
        return ids

    def add_xy_coords_to_hrrr_xarray(self, ids):
        """Function to add x and y in meters in native projection as coords"""
        xylocs = ccrs.CRS(self.hrrr_crs).transform_points(
            x=ids.coords['longitude'].values.ravel(),
            y=ids.coords['latitude'].values.ravel(),
            src_crs=self.geo_crs
        )
        hrrr_x1d = xylocs[:, 0].reshape(ids.dims['y'], ids.dims['x'])[0, :]
        hrrr_y1d = xylocs[:, 1].reshape(ids.dims['y'], ids.dims['x'])[:, 0]
        ids = ids.assign_coords(x=hrrr_x1d, y=hrrr_y1d)
        return ids

    def download_this_time(self, time_utc):
        """Function for downloading data at the given time"""
        success = True
        #time_str = np.datetime_as_string(time_utc, unit='h', timezone='UTC')
        #self.printit(f'Trying {time_str}..')
        try:
            hobj = HRRR(time_utc)
            ivar, ivarname = list(self.variables.items())[0]
            ids = hobj.get_xarray_for_regex(ivar, remove_grib=False)
            ids = self.add_xy_coords_to_hrrr_xarray(ids)
            ids = self.refine_hrrr_xarray(ids)
            ids = ids.rename({list(ids.keys())[0]: ivarname})
        except Exception as _:
            success = False
            #self.printit(f'{time_str}: aws or ssrs.HRRR issue')
        else:
            for ivar, iname in self.variables.items():
                try:
                    tds = hobj.get_xarray_for_regex(ivar, remove_grib=False)
                    idata = tds[list(tds.keys())[0]].values
                    if idata.shape != (ids.y.size, ids.x.size):
                        raise ValueError
                    ids[iname] = (('y', 'x'), idata)
                    ids[iname].attrs = tds[list(tds.keys())[0]].attrs
                except Exception as _:
                    success = False
                    #self.printit(f' {time_str}-{iname}-problem!')
            ids['time'] = self.time_utc
            ids = ids.expand_dims(dim='time')
            ids.to_netcdf(self.filepath)
        return success

    def download(self):
        """DOwnload function"""
        try:
            ids = xr.open_dataset(self.filepath, engine='scipy')
            if not set(self.variables.values()).issubset(set(ids.keys())):
                raise FileNotFoundError
        except Exception as _:
            success = False
            itr = iter(sorted(np.arange(-12, 12), key=abs))
            while not success:
                new_time_utc = self.time_utc + np.timedelta64(next(itr), 'h')
                success = self.download_this_time(new_time_utc)

    @classmethod
    def download_function(cls, ix):
        """Download Function"""
        with open(os.devnull, 'w', encoding='UTF-8') as f:
            time_utc, out_dir = ix
            #print(lonlat_bnd, domain_dir, proj_crs, resolution)
            #sys.stdout = f
            hobj = DataHRRR(
                time_utc=time_utc,
                out_dir=out_dir
            )
            hobj.download()
            #sys.stdout = sys.__stdout__
        return hobj

    @classmethod
    def annotate_function(cls, ituple):
        """Annotate function"""
        xlocs, ylocs, tlocs, _, fpath = ituple
        out_dict = {}
        try:
            ds = xr.open_mfdataset(
                fpath.as_posix(),
                combine='nested',
                concat_dim=('time')
            )
        except Exception as _:
            print(f'{fpath.as_posix()}:check-this-day')
        else:
            #print(f'{day_string}:got-{ds.time.size}-times', flush=True)
            for _, iname in DataHRRR.variables.items():
                #ds[iname] = ds[iname].interpolate_na(dim='time')
                out_dict[iname] = ds[iname].interp(
                    time=xr.DataArray(tlocs, dims=['points']),
                    x=xr.DataArray(xlocs, dims=['points']),
                    y=xr.DataArray(ylocs, dims=['points']),
                    method='linear',
                    kwargs={'fill_value': None},
                    # kwargs={'fill_value': 'extrapolate'}
                ).values.astype('float32')
        return out_dict
