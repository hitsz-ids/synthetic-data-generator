from __future__ import annotations

from pathlib import Path
from typing import Generator

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.extension import hookimpl
from sdgx.data_connectors.base import DataConnector
from sdgx.exceptions import CacheError
from sdgx.log import logger


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

    def clear_invalid_cache(self):
        """
        Clear all cache in cache_dir.
        """
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()

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
        data = data_connector.read(offset=offset, limit=max(self.blocksize, chunksize))
        self._refresh(offset, data)
        if len(data) < chunksize:
            return data
        return data[:chunksize]

    def load_all(self, data_connector: DataConnector) -> pd.DataFrame:
        """
        Load all data from data_connector or cache
        """
        return pd.concat(
            self.iter(chunksize=self.blocksize, data_connector=data_connector),
            ignore_index=True,
        )

    def iter(
        self, chunksize: int, data_connector: DataConnector
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Load data from data_connector or cache in chunk
        """
        offset = 0
        while True:
            data = self.load(offset, chunksize, data_connector)
            if len(data) == 0:
                break
            yield data
            offset += len(data)


@hookimpl
def register(manager):
    manager.register("DiskCache", DiskCache)
