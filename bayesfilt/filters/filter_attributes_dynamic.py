""" Base class for defining Bayesian filtering attributes """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name
from typing import Callable
from functools import partial
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy import ndarray


@dataclass
class FilterAttributesDynamic:
    """Dynamic attributes of a filter"""
    _y: ndarray | None = None
    _m: ndarray | None = None
    _P: ndarray | None = None
    _R: ndarray | None = None
    _time_elapsed: float | None = None
    _last_update_at: float | None = None
    cur_metrics: dict[str, float] = field(default_factory=dict, repr=True)
    cur_smetrics: dict[str, float] = field(default_factory=dict)
    _dfraw: dict = field(default_factory=dict, repr=False)
    mean_colname: str = field(default='StateMean', repr=False)
    cov_colname: str = field(default='StateCov', repr=False)
    time_colname: str = field(default='TimeElapsed', repr=False)
    metrics_colname: str = field(default='Metrics', repr=False)
    y_colname: str = field(default='Observation', repr=False)
    ycov_colname: str = field(default='ObservationCov', repr=False)
    smean_colname: str = field(default='StateMeanSmoother', repr=False)
    scov_colname: str = field(default='StateCovSmoother', repr=False)
    smetrics_colname: str = field(default='MetricsSmoother', repr=False)

    def _add_new_entry(self):
        new_entry = {
            self.time_colname: np.around(self.time_elapsed, 4),
            self.y_colname: deepcopy(self.y),
            self.mean_colname: deepcopy(self.m),
            self.cov_colname: deepcopy(self.P),
            self.ycov_colname: deepcopy(self.R),
            self.metrics_colname: self.cur_metrics
        }
        for k, v in new_entry.items():
            if k in self._dfraw:
                self._dfraw[k].append(v)
            else:
                self._dfraw[k] = [v]

    def store_this_timestep(self, update: bool = False) -> None:
        """Store this forecast/update step"""
        #self._time_elapsed = np.around(self.time_elapsed, 3)
        if update is True:
            for _, v in self._dfraw.items():
                del v[-1]
            self._last_update_at = self._time_elapsed
        self._add_new_entry()

    @property
    def dfraw(self) -> pd.DataFrame:
        """State mean"""
        return pd.DataFrame(self._dfraw)
        # return self._dfraw

    @property
    def y(self) -> ndarray:
        """Observation vector"""
        return self._y

    @property
    def m(self) -> ndarray:
        """State mean vector"""
        return self._m

    @property
    def P(self) -> ndarray:
        """State covariance matrix"""
        return self._P

    @property
    def R(self) -> ndarray:
        """Observation error covariance matrix"""
        return self._R

    @property
    def time_elapsed(self) -> float:
        """Time elapsed"""
        return self._time_elapsed

    @property
    def last_update_at(self) -> float:
        """Time of last update"""
        return self._last_update_at

    def raiseit(self, outstr: str = "") -> None:
        """Raise exception with the out string"""
        raise ValueError(f'{self.__class__.__name__}: {outstr}')

    @property
    def lifespan_to_last_update(self):
        """Returns the time duration of existence till the last update"""
        return self.last_update_at - self._dfraw[self.time_colname].iloc[0]

    @property
    def tlist(self) -> ndarray:
        """Get list of time elapsed"""
        return self.dfraw[self.time_colname].values

    # @property
    # def is_ready_for_smoother(self):
    #     """Return true if ready for smoother"""
    #     out_bool = False
    #     if len(self.dfraw[self.time_colname]) > 1:
    #         print('No state history found, run filter() first!')
    #     else:
    #         out_bool = True
    #     return out_bool

    # @property
    # def metrics_dfraw(self) -> ndarray:
    #     """Get pandas df for metrics"""
    #     return pd.DataFrame(self.dfraw[self.metrics_colname].values)
