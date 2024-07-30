from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Generator

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.extension import hookimpl
from sdgx.data_connectors.base import DataConnector
from sdgx.exceptions import CacheError
from sdgx.utils import logger


class DiskCache(Cacher):
    """
    Cacher that cache data in disk with parquet format

    Args:
        blocksize (int): The blocksize of the cache.
        cache_dir (str | Path | None, optional): The directory where the cache will be stored. Defaults to None.
        identity (str | None, optional): The identity of the data source. Defaults to None.

    Todo:
        * Add partial cache when blocksize > chunksize
        * Improve cache invalidation
        * Improve performance if blocksize > chunksize
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        identity: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not cache_dir:
            cache_dir = Path.cwd() / ".sdgx_cache"
            if identity:
                cache_dir = cache_dir / identity
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def clear_cache(self) -> None:
        """
        Clear all cache in cache_dir.
        """
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def clear_invalid_cache(self):
        """
        Clear all cache in cache_dir.

        TODO: Improve cache invalidation
        """
        return self.clear_cache()

    def _get_cache_filename(self, offset: int) -> Path:
        """
        Get cache filename
        """
        return self.cache_dir / f"{offset}.parquet"

    def is_cached(self, offset: int) -> bool:
        """
        Check if the data is cached by checking if the cache file exists
        """
        return self._get_cache_filename(offset).exists()

    def _refresh(self, offset: int, data: pd.DataFrame) -> None:
        """
        Refresh cache, will write data to cache file in parquet format.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if len(data) < self.blocksize:
            data.to_parquet(self._get_cache_filename(offset))
        elif len(data) > self.blocksize:
            for i in range(0, len(data), self.blocksize):
                data[i : i + self.blocksize].to_parquet(self._get_cache_filename(offset + i))
        else:
            data.to_parquet(self._get_cache_filename(offset))

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
        """
        Load data from data_connector or cache
        """

        if chunksize % self.blocksize != 0:
            raise CacheError(
                "chunksize must be multiple of blocksize, current chunksize is {} and blocksize is {}".format(
                    chunksize, self.blocksize
                )
            )

        if chunksize != self.blocksize:
            logger.warning("chunksize must be equal to blocksize, may cause performance issue.")
        if self.is_cached(offset):
            cached_data = pd.read_parquet(self._get_cache_filename(offset))
            if len(cached_data) >= chunksize:
                return cached_data[:chunksize]
            return cached_data
        limit = max(self.blocksize, chunksize)
        data = data_connector.read(offset=offset, limit=limit)
        if data is None:
            return data
        while len(data) < limit:
            # When generator size is less than blocksize
            # Continue to read until fit the limit
            next_data = data_connector.read(offset=offset + len(data), limit=limit - len(data))
            if next_data is None or len(next_data) == 0:
                break
            data = pd.concat(
                [
                    data,
                    next_data,
                ],
                ignore_index=True,
            )

        self._refresh(offset, data)
        if len(data) < chunksize:
            return data
        return data[:chunksize]

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load data from data_connector or cache in chunk
        """
        offset = 0
        while True:
            data = self.load(offset, chunksize, data_connector)
            if data is None or len(data) == 0:
                break
            yield data
            offset += len(data)


@hookimpl
def register(manager):
    manager.register("DiskCache", DiskCache)
