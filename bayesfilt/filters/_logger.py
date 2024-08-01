""" Base class for defining logger needed for Bayesian filtering """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name

from dataclasses import dataclass, field
from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import ndarray


@dataclass
class FilterLogger:
    """Logger class"""
    _raw: list = field(default_factory=list, repr=False)

    @property
    def df(self):
        """return raw data as pandas dataframe"""
        idf = pd.DataFrame(self._raw)
        if not idf.empty:
            idf.flag = idf.flag.astype('category')
        return idf

    def reset(self):
        """reset the logger"""
        self._raw = []

    def reverse(self):
        """reverse the order"""
        self._raw = list(reversed(self._raw))

    def record(
        self,
        time_elapsed: float,
        state_mean: ndarray,
        state_cov: ndarray | None = None,
        obs: ndarray | None = None,
        obs_cov: ndarray | None = None,
        metrics: dict | None = None,
        flag: str = 'None'
    ):
        """add to the logger"""
        new_entry = dict(
            time_elapsed=time_elapsed,
            state_mean=deepcopy(state_mean),
            state_cov=deepcopy(state_cov),
            obs=deepcopy(obs),
            obs_cov=deepcopy(obs_cov),
            metrics=deepcopy(metrics),
            flag=flag
        )
        self._raw.append(new_entry)

    # def is_flag(self, flag: str) -> ndarray:
    #     """Get state mean for idx state index"""
    #     if not self.df.empty:
    #         return self.df.flag == flag

    def state_mean(self, x_idx: int | None = None) -> ndarray | None:
        """Get x_idx state for all times """
        if not self.df.empty:
            if x_idx is not None:
                return np.stack(self.df.state_mean.values)[:, x_idx]
            else:
                return np.stack(self.df.state_mean.values)

    def state_var(
        self,
        x1_idx: int | None = None,
        x2_idx: int | None = None
    ) -> ndarray | None:
        """Get entry (idx_1, x2_idx) from the cov matrix"""
        if not self.df.empty:
            if x1_idx is not None:
                x2_idx = x1_idx if x2_idx is None else x2_idx
                return np.stack(self.df.state_cov.values)[:, x1_idx, x2_idx]
            else:
                return np.stack(self.df.state_cov.values)

    def obs(self, t_idx) -> ndarray | None:
        """Get observation vector at given time index"""
        if not self.df.empty:
            return self.df.obs.iloc[t_idx]

    def obs_var(self, t_idx: int) -> ndarray | None:
        """Get observation covariance matrix at t_idx time index"""
        if not self.df.empty:
            return self.df.obs_cov.iloc[t_idx]

    def flag(self, t_idx: int | None = None) -> ndarray | None:
        """Get observation covariance matrix at t_idx time index"""
        if not self.df.empty:
            if t_idx is not None:
                return self.df.flag.iloc[t_idx]
            else:
                return self.df.flag.values

    @property
    def time_elapsed(self):
        """Get state mean for idx state index"""
        if not self.df.empty:
            return np.stack(self.df.time_elapsed.values)
