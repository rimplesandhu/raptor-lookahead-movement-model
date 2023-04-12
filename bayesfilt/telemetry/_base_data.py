""" Base class for defining data classes"""
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
import os
import sys
from functools import partial
import logging
from pathlib import Path
from tqdm import tqdm
import cartopy.crs as ccrs


class BaseClass:
    """Base class for telemetry module"""

    def __init__(
        self,
        out_dir: str | None = None,
        log_level: int | None = None
    ):

        # logging
        logging.basicConfig(
            level=log_level,
            format='%(name)-12s:%(levelname)-s: %(message)s',
        )
        self.log = logging.getLogger(self.__class__.__name__)
        log_level = 40 if log_level is None else log_level
        self.log.setLevel(log_level)

        # directories
        self.out_dir = Path(os.path.join(os.getcwd(), 'output'))
        if out_dir is not None:
            self.out_dir = Path(out_dir)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)

    @property
    def pbar(self):
        """Returns progress bar"""
        return partial(tqdm, desc=self.__class__.__name__,
                       position=0, leave=True, file=sys.stdout)

    def raiseit(self, outstr: str = "", err=ValueError) -> None:
        """Raise exception with the out string"""
        raise err(f'{self.__class__.__name__}: {outstr}')

    def printit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        print(f'{self.__class__.__name__}: {outstr}', flush=True)


class BaseGeoData(BaseClass):
    """Base class for defining varous dataclasses"""

    def __init__(
        self,
        proj_crs: str | None = None,
        **kwargs
    ):
        BaseClass.__init__(self, **kwargs)
        if proj_crs is not None:
            if isinstance(proj_crs, str):
                self.proj_crs = proj_crs
            elif isinstance(proj_crs, ccrs.CRS):
                self.proj_crs = proj_crs
            else:
                self.raiseit('Invalid CRS input!')
        else:
            self.proj_crs = self.albersna_crs

    @property
    def albersna_crs(self):
        """Albers equal area conic projection"""
        return ccrs.CRS('ESRI:102008')

    @property
    def geo_crs(self):
        """Returns geographical ref system"""
        return ccrs.CRS('EPSG:4326')
