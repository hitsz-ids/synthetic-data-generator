from typing import Any, Generator

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.manager import CacherManager
from sdgx.data_connectors.base import DataConnector


class DataLoader:
    """
    Wrapper of :ref:`DataConnector`
    """

    def __init__(
        self,
        data_connector: DataConnector,
        chunksize: int = 1000,
        cacher: Cacher | None = None,
        cache_mode: str = "MemoryCache",
        cacher_kwargs: None | dict[str, Any] = None,
    ) -> None:
        self.data_connector = data_connector
        self.chunksize = chunksize
        self.cache_manager = CacherManager()

        cacher_kwargs.setdefault("blocksize", self.chunksize)
        if not cacher:
            self.cacher = self.cache_manager.init_cacher(cache_mode, **cacher_kwargs)
        self.cacher = cacher

    def iter(self) -> Generator[pd.DataFrame, None, None]:
        for d in self.cacher.iter(self.chunksize, self.data_connector):
            yield d

    def keys(self) -> list:
        return self.data_connector.keys()

    def columns(self) -> list:
        return self.data_connector.columns()

    def load_all(self) -> pd.DataFrame:
        return self.cacher.load_all(self.data_connector)
