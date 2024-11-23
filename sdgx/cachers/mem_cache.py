from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Generator, List, Dict

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.extension import hookimpl
from sdgx.data_connectors.base import DataConnector
from sdgx.exceptions import CacheError
from sdgx.utils import logger

# TODO 待完成
# class MemCache(Cacher):
#     def __init__(
#             self,
#             identity: str | None = None,
#             *args,
#             **kwargs,
#     ) -> None:
#         super().__init__(*args, **kwargs)
#         self.identity: str = identity
#         self.__lock = False
#         self.offset_mapper: Dict[int, io.BytesIO] = {}
#
#     def clear_cache(self) -> None:
#         """
#         Clear all cache in cache_dir.
#         """
#         self.__lock = True
#         for i in self.offset_mapper.values():
#             i.close()
#         self.offset_mapper = {}
#         self.__lock = False
#
#     def clear_invalid_cache(self):
#         """
#         Clear all cache in cache_dir.
#
#         TODO: Improve cache invalidation
#         """
#         return self.clear_cache()
#
#     def _get_cache_filename(self, offset: int) -> int:
#         """
#         Get cache filename
#         """
#         return offset
#
#     def is_cached(self, offset: int) -> bool:
#         """
#         Check if the data is cached by checking if the cache file exists
#         """
#         return self._get_cache_filename(offset) in self.offset_mapper
#
#     def _refresh(self, offset: int, data: pd.DataFrame) -> None:
#         """
#         Refresh cache, will write data to cache file in parquet format.
#         """
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         if len(data) < self.blocksize:
#             data.to_parquet(self._get_cache_filename(offset))
#         elif len(data) > self.blocksize:
#             for i in range(0, len(data), self.blocksize):
#                 data[i: i + self.blocksize].to_parquet(self._get_cache_filename(offset + i))
#         else:
#             data.to_parquet(self._get_cache_filename(offset))
#
#     def load(self, offset: int, chunksize: int, data_connector: DataConnector) -> pd.DataFrame | None:
#         """
#         Load data from data_connector or cache
#         """
#
#         if chunksize % self.blocksize != 0:
#             raise CacheError(
#                 "chunksize must be multiple of blocksize, current chunksize is {} and blocksize is {}".format(
#                     chunksize, self.blocksize
#                 )
#             )
#
#         if chunksize != self.blocksize:
#             logger.warning("chunksize must be equal to blocksize, may cause performance issue.")
#
#         if self.is_cached(offset):
#             cached_data = pd.read_parquet(self._get_cache_filename(offset))
#             if len(cached_data) >= chunksize:
#                 return cached_data[:chunksize]
#             return cached_data
#         limit = max(self.blocksize, chunksize)
#         data = data_connector.read(offset=offset, limit=limit)
#         if data is None:
#             return data
#
#         data_list: List[pd.DataFrame] = [data]
#         while len(data) < limit:
#             # When generator size is less than blocksize
#             # Continue to read until fit the limit
#             next_data = data_connector.read(offset=offset + len(data), limit=limit - len(data))
#             if next_data is None or len(next_data) == 0:
#                 break
#             data_list.append(next_data)
#         data = pd.concat(
#             data_list,
#             ignore_index=True,
#         )
#
#         self._refresh(offset, data)
#         if len(data) < chunksize:
#             return data
#         return data[:chunksize]
#
#     def iter(
#             self, chunksize: int, data_connector: DataConnector
#     ) -> Generator[pd.DataFrame, None, None]:
#         """
#         Load data from data_connector or cache in chunk
#         """
#         offset = 0
#         while True:
#             data = self.load(offset, chunksize, data_connector)
#             if data is None or len(data) == 0:
#                 break
#             yield data
#             offset += len(data)
#
#
# @hookimpl
# def register(manager):
#     manager.register("MemCache", MemCache)
