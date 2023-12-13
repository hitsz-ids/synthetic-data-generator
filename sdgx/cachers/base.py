from __future__ import annotations

from typing import Generator

import pandas as pd

from sdgx.cachers.extension import hookimpl
from sdgx.data_connectors.base import DataConnector


class Cacher:
    def __init__(self, blocksize, *args, **kwargs) -> None:
        self.blocksize = blocksize

    def is_cached(self, offset: int) -> bool:
        raise NotImplementedError

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
        raise NotImplementedError

    def load_all(self, data_connector: DataConnector) -> pd.DataFrame:
        raise NotImplementedError

    def clear_invalid_cache(self):
        return

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError


class NoCache(Cacher):
    def is_cached(self, offset: int) -> bool:
        return False

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
        return data_connector.read(offset=offset, limit=chunksize)

    def load_all(self, data_connector: DataConnector) -> pd.DataFrame:
        return data_connector.read(offset=0, limit=None)

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        for d in data_connector.iter(chunksize=chunksize):
            yield d


@hookimpl
def register(manager):
    manager.register("NoCache", NoCache)
