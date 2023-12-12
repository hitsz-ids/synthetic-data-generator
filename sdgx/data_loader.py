from __future__ import annotations

from typing import Any, Generator

import pandas as pd

from sdgx.data_connectors.base import DataConnector
from sdgx.exceptions import CacheError
from sdgx.log import logger


class Cacher:
    def __init__(self, blocksize, *args, **kwargs) -> None:
        self.blocksize = blocksize

    def is_cached(self, offset: int) -> bool:
        raise NotImplementedError

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.Dataframe:
        raise NotImplementedError

    def load_all(self, data_connector: DataConnector) -> pd.Dataframe:
        raise NotImplementedError

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError


class NoCache(Cacher):
    def is_cached(self, offset: int) -> bool:
        return False

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.Dataframe:
        return data_connector.read(offset=offset, limit=chunksize)

    def load_all(self, data_connector: DataConnector) -> pd.Dataframe:
        return data_connector.read(offset=0, limit=None)

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        for d in data_connector.iter(chunksize=chunksize):
            yield d


class MemoryCache(Cacher):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache = {}

    def is_cached(self, offset: int) -> bool:
        return offset in self.cache

    def _refresh(self, offset: int, data: pd.DataFrame) -> None:
        if len(data) < self.blocksize:
            self.cache[offset] = data
        elif len(data) > self.blocksize:
            for i in range(0, len(data), self.blocksize):
                self.cache[offset + i] = data[i : i + self.blocksize]
        else:
            self.cache[offset] = data

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.Dataframe:
        if chunksize % self.blocksize != 0:
            raise CacheError(
                "chunksize must be multiple of blocksize, current chunksize is {} and blocksize is {}".format(
                    chunksize, self.blocksize
                )
            )

        if offset in self.cache:
            cached_data = self.cache[offset]
            if len(cached_data) >= chunksize:
                return cached_data[:chunksize]
            return cached_data

        data = data_connector.read(offset=offset, limit=max(self.blocksize, chunksize))
        self._refresh(offset, data)
        if len(data) < chunksize:
            return data
        return data[:chunksize]

    def load_all(self, data_connector: DataConnector) -> pd.Dataframe:
        # Concat all dataframe
        return pd.concat(
            self.iter(chunksize=self.blocksize, data_connector=data_connector),
            ignore_index=True,
        )

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        offset = 0
        while True:
            data = self.load(offset, chunksize, data_connector)
            if len(data) == 0:
                break
            yield data
            offset += len(data)


class DiskCache(Cacher):
    ...


class DataLoader:
    """
    Wrapper of :ref:`DataConnector`
    """

    support_cache_mode = {
        "no_cache": NoCache,
        "memory": MemoryCache,
        "disk": DiskCache,
    }
    default_cacher_cls = MemoryCache

    def __init__(
        self,
        data_connector: DataConnector,
        chunksize: int = 1000,
        cacher: Cacher | None = None,
        cache_mode: str = "memory",
        cacher_kwargs: None | dict[str, Any] = None,
    ) -> None:
        self.data_connector = data_connector
        self.chunksize = chunksize

        if not cacher:
            try:
                self.cacher = self.support_cache_mode[cache_mode](**(cacher_kwargs or {}))
            except KeyError:
                raise NotImplementedError("Not support cache mode")
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
