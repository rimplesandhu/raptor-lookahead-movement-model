""" Base class for telemetry data """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
# pylint: disable=logging-fstring-interpolation
from typing import Callable
from functools import partial
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
from matplotlib import patches
import matplotlib.pyplot as plt
from numpy import ndarray
from .utils import get_bin_edges
from .telemetry import Telemetry


@dataclass
class TelemetryPlotter(Telemetry):
    """Plotting utility class for Telemetry class"""

    def plot_subdomain(
        self,
        idomain: str,
        *args,
        ax: plt.Axes | None = None,
        **kwargs
    ):
        """Plots the data in this subdomain"""
        if self.df_subdomains.empty:
            self.printit('No subdomain info found!')
        else:
            valid_domains = self.df_subdomains.index.tolist()
            assert idomain in valid_domains, f'{idomain} invalid subdomain!'
            xy_bounds = self.df_subdomains.loc[idomain].values
            domain_df = self.df[self.df[self.domain_col] == idomain]
            ax = plt.gca() if ax is None else ax
            domain_box = patches.Rectangle(
                xy=(xy_bounds[0], xy_bounds[2]),
                width=xy_bounds[1] - xy_bounds[0],
                height=xy_bounds[3] - xy_bounds[2],
                fill=True, alpha=0.1
            )
            ax.add_artist(domain_box)
            ax.plot(domain_df[self.x_col],
                    domain_df[self.y_col],
                    *args, **kwargs)
            ax.set_aspect('equal')
            # ax.set_xlim(*xy_bounds[:2])

    def plot_track_in_time(
        self,
        track_id: int,
        col_name: str,
        *args,
        ax: plt.Axes | None = None,
        **kwargs
    ):
        """Plot this tracks"""
        ax = plt.gca() if ax is None else ax
        self.check_validity_of_columns(col_name)
        dftrack = self.df[self.df[self.trackid_col] == track_id]
        tlist = dftrack[self.tracktime_col]
        cb = ax.plot(tlist, dftrack[col_name], *args, **kwargs)
        ax.set_ylabel(f'{col_name}')
        ax.set_xlim([tlist.min(), tlist.max()])
        ax.set_xlabel('Time elapsed (Seconds)')
        return cb

    def plot_track_in_space(
        self,
        track_id: int,
        *args,
        ax: plt.Axes | None = None,
        **kwargs
    ):
        """Plot this tracks"""
        ax = plt.gca() if ax is None else ax
        dftrack = self.df[self.df[self.trackid_col] == track_id]
        cb = ax.plot(dftrack[self.x_col], dftrack[self.y_col], *args, **kwargs)
        ax.set_xlabel(f'{self.x_col}')
        ax.set_ylabel(f'{self.y_col}')
        return cb
