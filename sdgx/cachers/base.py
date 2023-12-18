from __future__ import annotations

from typing import Generator

import pandas as pd

from sdgx.cachers.extension import hookimpl
from sdgx.data_connectors.base import DataConnector


class Cacher:
    """
    Base class for cachers

    Cacher is used to cache raw data and processed data to prevent repeat read or process.

    You can treat Cacher as a warrper of :ref:`DataConnector`
    """

    def __init__(self, blocksize, *args, **kwargs) -> None:
        self.blocksize = blocksize

    def is_cached(self, offset: int) -> bool:
        """
        Check if the data is cached
        """

        raise NotImplementedError

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
        """
        Load data from data_connector or cache
        """
        raise NotImplementedError

    def load_all(self, data_connector: DataConnector) -> pd.DataFrame:
        """
        Load all data from data_connector or cache
        """
        return pd.concat(
            self.iter(chunksize=self.blocksize, data_connector=data_connector),
            ignore_index=True,
        )

    def clear_cache(self):
        """
        Clear all cache
        """
        return

    def clear_invalid_cache(self):
        """
        Clear invalid cache.

        It's useful when data source has been changed.
        Subclass can try to inspect cache and only clear invalid cache.
        Also, it may clear all cache when not sure or not support.
        """
        return

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load data from data_connector or cache in chunk
        """

        raise NotImplementedError


class NoCache(Cacher):
    """
    No cache means to proxy data_connector
    """

    def is_cached(self, offset: int) -> bool:
        """
        Always return False
        """
        return False

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
        """
        Proxy to data_connector.read
        """
        return data_connector.read(offset=offset, limit=chunksize)

    def load_all(self, data_connector: DataConnector) -> pd.DataFrame:
        """
        Proxy to data_connector.read
        """
        return data_connector.read(offset=0, limit=None)

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Proxy to data_connector.iter
        """
        for d in data_connector.iter(chunksize=chunksize):
            yield d


@hookimpl
def register(manager):
    manager.register("NoCache", NoCache)
