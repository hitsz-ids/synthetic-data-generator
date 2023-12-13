from pathlib import Path
from typing import Generator

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.extension import hookimpl
from sdgx.data_connectors.base import DataConnector
from sdgx.exceptions import CacheError
from sdgx.log import logger


class DiskCache(Cacher):
    def __init__(self, cache_dir: str | Path = Path("/tmp/sdgx/"), *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_filename(self, offset: int) -> Path:
        return self.cache_dir / f"{offset}.parquet"

    def is_cached(self, offset: int) -> bool:
        return self._get_cache_filename(offset).exists()

    def _refresh(self, offset: int, data: pd.DataFrame) -> None:
        if len(data) < self.blocksize:
            data.to_parquet(self._get_cache_filename(offset))
        elif len(data) > self.blocksize:
            for i in range(0, len(data), self.blocksize):
                data[i : i + self.blocksize].to_parquet(self._get_cache_filename(offset + i))
        else:
            data.to_parquet(self._get_cache_filename(offset))

    def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame:
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


@hookimpl
def register(manager):
    manager.register("DiskCache", DiskCache)
