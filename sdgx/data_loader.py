from __future__ import annotations

from functools import cached_property
from typing import Any, Generator

import pandas as pd

from sdgx.cachers.base import Cacher
from sdgx.cachers.manager import CacherManager
from sdgx.data_connectors.base import DataConnector
from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.utils import cache


class DataLoader:
    """
    Combine :ref:`Cacher` and :ref:`DataConnector` to load data in an efficient way.

    Default Cacher is :ref:`DiskCache`. Use ``cacher`` or ``cache_mode`` to specify a :ref:`Cacher`.

    Args:
        data_connector (:ref:`DataConnector`): The data connector
        chunksize (int, optional): The chunksize of the cacher. Defaults to 1000.
        cacher (:ref:`Cacher`, optional): The cacher. Defaults to None.
        cache_mode (str, optional): The cache mode(cachers' name). Defaults to "DiskCache", more info in :ref:`DiskCache`.
        cacher_kwargs (dict, optional): The kwargs for cacher. Defaults to None
    """

    def __init__(
        self,
        data_connector: DataConnector,
        chunksize: int = 10000,
        cacher: Cacher | None = None,
        cache_mode: str = "DiskCache",
        cacher_kwargs: None | dict[str, Any] = None,
    ) -> None:
        self.data_connector = data_connector
        self.chunksize = chunksize
        self.cache_manager = CacherManager()

        if not cacher_kwargs:
            cacher_kwargs = {}
        cacher_kwargs.setdefault("blocksize", self.chunksize)
        cacher_kwargs.setdefault("identity", self.data_connector.identity)
        self.cacher = cacher or self.cache_manager.init_cacher(cache_mode, **cacher_kwargs)

        self.cacher.clear_invalid_cache()

        if isinstance(data_connector, GeneratorConnector):
            # Warmup cache
            self.load_all()

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

    def finalize(self, clear_cache=False) -> None:
        """
        Finalize the dataloader.
        """
        self.data_connector.finalize()
        if clear_cache:
            self.cacher.clear_cache()

    def __getitem__(self, key: list | slice | tuple) -> pd.DataFrame:
        """
        Support get data by index and slice.

        We will warmup cache for GeneratorConnector so that we can randomly access data.

        """
        if isinstance(key, list):
            sli = None
            rows = key
        else:
            sli = key
            rows = None

        if not sli:
            return pd.concat((d[rows] for d in self.iter()), ignore_index=True)

        start = sli.start or 0
        stop = sli.stop or len(self)
        step = sli.step or 1

        offset = (start // self.chunksize) * self.chunksize
        n_iter = ((stop - start) // self.chunksize) + 1

        tables = (
            self.cacher.load(
                offset=offset + i * self.chunksize,
                chunksize=self.chunksize,
                data_connector=self.data_connector,
            )
            for i in range(n_iter)
        )

        return pd.concat(tables, ignore_index=True)[start - offset : stop - offset : step]

    @cache
    def __len__(self):
        return sum(len(l) for l in self.iter())

    @cached_property
    def shape(self):
        return (len(self), len(self.columns()))
