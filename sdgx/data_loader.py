from typing import Any, Generator

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.manager import CacherManager
from sdgx.data_connectors.base import DataConnector


class DataLoader:
    """
    Combine :ref:`Cacher` and :ref:`DataConnector` to load data in an efficient way.

    Default Cacher is :ref:`DiskCache`. Use ``cacher`` or ``cache_mode`` to specify a :ref:`Cacher`.

    Args:
        data_connector (:ref:`DataConnector`): The data connector
        chunksize (int, optional): The chunksize of the cacher. Defaults to 1000.
        cacher (:ref:`Cacher`, optional): The cacher. Defaults to None.
        cache_mode (str, optional): The cache mode(name). Defaults to "DiskCache".
        cacher_kwargs (dict, optional): The kwargs for cacher. Defaults to None
    """

    def __init__(
        self,
        data_connector: DataConnector,
        chunksize: int = 1000,
        cacher: Cacher | None = None,
        cache_mode: str = "DiskCache",
        cacher_kwargs: None | dict[str, Any] = None,
    ) -> None:
        self.data_connector = data_connector
        self.chunksize = chunksize
        self.cache_manager = CacherManager()

        cacher_kwargs.setdefault("blocksize", self.chunksize)
        cacher_kwargs.setdefault("identity", self.data_connector.identity)
        if not cacher:
            self.cacher = self.cache_manager.init_cacher(cache_mode, **cacher_kwargs)
        self.cacher = cacher

        self.cacher.clear_invalid_cache()

    def iter(self) -> Generator[pd.DataFrame, None, None]:
        """
        Load data from cache in chunk.
        """
        for d in self.cacher.iter(self.chunksize, self.data_connector):
            yield d

    def keys(self) -> list:
        """
        Same as ``columns``
        """
        return self.data_connector.keys()

    def columns(self) -> list:
        """
        Peak columns.

        Returns:
            list: name of columns
        """
        return self.data_connector.columns()

    def load_all(self) -> pd.DataFrame:
        """
        Load all data from cache.
        """
        return self.cacher.load_all(self.data_connector)
