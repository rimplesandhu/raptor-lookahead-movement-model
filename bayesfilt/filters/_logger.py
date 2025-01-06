""" Base class for defining logger needed for Bayesian filtering """
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-public-methods
# pylint: disable=invalid-name

from copy import deepcopy
import pandas as pd
import numpy as np
from numpy import ndarray


class FilterLogger:
    """Logger class"""

    def __init__(self):
        """initialization"""
        self._raw: list = []

    @property
    def dfraw(self):
        """return raw data as pandas dataframe"""
        return pd.DataFrame(self._raw)

    @property
    def is_initiated(self):
        """Check if data exists"""
        return self._raw

    def reset(self):
        """reset the logger"""
        self._raw = []

    def reverse(self):
        """reverse the order"""
        self._raw = list(reversed(self._raw))

    @property
    def last_update_index(self):
        if self.is_initiated:
            if self.dfraw.obs.notna().values.any():
                return self.dfraw.obs.notna()[::-1].idxmax()
            else:
                return 0  # when there is no update after start

    def get_df(
        self,
        xnames: list[str] | None = None,
        variance: bool = True,
        metrics: bool = True,
        # remove_forecast_at_update: bool = False,
        remove_beyond_last_update: bool = True
    ):
        """Assemble the dataframe from the logger"""

        if self.is_initiated:
            nstates = self.mean_at_step(0).size
            if xnames is None:
                xnames = [f'X{i}' for i in range(nstates)]
            else:
                assert len(xnames) == nstates, 'len(xnames) incompatible!'

            # basics
            idf = pd.DataFrame([])
            rdf = self.dfraw
            idf['TimeElapsed'] = rdf['time_elapsed']
            idf['Flag'] = rdf['flag']

            # # state mean and var
            state_mean = np.stack(rdf['state_mean'])
            state_cov = np.stack(rdf['state_cov'])
            for ix, iname in enumerate(xnames):
                idf[iname] = state_mean[:, ix]
                if variance:
                    idf[f'{iname}_Var'] = state_cov[:, ix, ix]

            # metrics
            if metrics:
                mdf = rdf['metrics'].apply(pd.Series)
                idf = pd.concat([idf, mdf], axis=1)

            # # # forecast at update step
            # # if remove_forecast_at_update:
            # #     ibool = idf.TimeElapsed.diff().shift(-1) == 0
            # #     ibool = (ibool) & (self.dfraw.obs.isnull())
            # #     idf.drop(idf[ibool].index, inplace=True)

            if remove_beyond_last_update:
                ibool = idf.index > self.last_update_index
                idf.drop(idf[ibool].index, inplace=True)
        return idf

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

    def remove_last_entry(self):
        """Removes last entry"""
        del self._raw[-1]

    def state_var_ij(
        self,
        x1_idx: int,
        x2_idx: int | None = None
    ) -> ndarray | None:
        """Get entry (idx_1, x2_idx) from the cov matrix"""
        if self.is_initiated:
            x2_idx = x1_idx if x2_idx is None else x2_idx
            assert isinstance(x1_idx, int), 'Index needs to be int!'
            assert isinstance(x2_idx, int), 'Index needs to be int!'
            return np.stack(self.dfraw.state_cov.values)[:, x1_idx, x2_idx]

    def time_at_step(self, idx: int) -> ndarray | None:
        """Get state mean for idx time step"""
        return self._raw[idx]['time_elapsed']

    def mean_at_step(self, idx: int) -> ndarray | None:
        """Get state mean for idx time step"""
        return self._raw[idx]['state_mean']

    def cov_at_step(self, idx: int):
        """Get state covariance for idx time step"""
        return self._raw[idx]['state_cov']

    def obs_at_step(self, idx) -> ndarray | None:
        """Get observation vector at given time index"""
        return self._raw[idx]['obs']

    def obs_var_at_step(self, idx: int) -> ndarray | None:
        """Get observation covariance matrix at t_idx time index"""
        return self._raw[idx]['obs_cov']

    def flag_at_step(self, idx: int) -> ndarray | None:
        """Get observation covariance matrix at t_idx time index"""
        return self._raw[idx]['flag']
