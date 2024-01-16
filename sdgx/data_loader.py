from __future__ import annotations

from functools import cached_property
from typing import Any, Generator

import pandas as pd

from sdgx.cachers.base import Cacher, NoCache
from sdgx.cachers.disk_cache import DiskCache
from sdgx.cachers.manager import CacherManager
from sdgx.data_connectors.base import DataConnector
from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.exceptions import DataLoaderInitError
from sdgx.utils import cache


class DataLoader:
    """
    Combine :ref:`Cacher` and :ref:`DataConnector` to load data in an efficient way.

    Default Cacher is :ref:`DiskCache`. Use ``cacher`` or ``cache_mode`` to specify a :ref:`Cacher`.

    GeneratorConnector must combine with Cacher, we will warmup cache for generator to support random access.

    Args:
        data_connector (:ref:`DataConnector`): The data connector
        chunksize (int, optional): The chunksize of the cacher. Defaults to 1000.
        cacher (:ref:`Cacher`, optional): The cacher. Defaults to None.
        cache_mode (str, optional): The cache mode(cachers' name). Defaults to "DiskCache", more info in :ref:`DiskCache`.
        cacher_kwargs (dict, optional): The kwargs for cacher. Defaults to None
        identity (str, optional): The identity of the data source.
            When using :ref:`GeneratorConnector`, it can be pointed to the original data source, makes it possible to work with :ref:`MetadataCombiner`.

    Example:

        Load and cache data from existing csv file or other data source.

        .. code-block:: python

            from sdgx.data_loader import DataLoader
            from sdgx.data_connectors.csv_connector import CsvConnector
            from sdgx.utils import download_demo_data

            dataset_csv = download_demo_data()
            data_connector = CsvConnector(path=dataset_csv)

            # Use DataConnector to initialize

            dataloader = DataLoader(data_connector)

            # Access data

            dataloader.load_all()  # This will read all data from csv, and cache it.
            dataloader.load_all()  # This will read all data from cache.

            dataloader[:10] # dataloader support slicing

            for df in dataloader.iter():  # dataloader support iteration
                print(df.shape)

    Advanced usage:

        Load and cache data from a generator.

        .. code-block:: python

            from sdgx.data_loader import DataLoader
            from sdgx.data_connectors.generator_connector import GeneratorConnector

            def generator() -> Generator[pd.DataFrame, None, None]:
                for i in range(100):
                    yield pd.DataFrame({"a": [i], "b": [i]})

            data_connector = GeneratorConnector(generator)

            # Use DataConnector to initialize.
            # Generator is not support random access, but we can achieve it by caching.
            dataloader = DataLoader(data_connector)

            # Access data
            dataloader.load_all()  # This will read all data from cache
            dataloader.load_all()  # This will read all data from cache.

            dataloader[:10] # dataloader support slicing

            for df in dataloader.iter():  # dataloader support iteration
                print(df.shape)


    """

    DEFAULT_CACHER = DiskCache

    def __init__(
        self,
        data_connector: DataConnector,
        chunksize: int = 10000,
        cacher: Cacher | str | type[Cacher] | None = None,
        cacher_kwargs: None | dict[str, Any] = None,
        identity: str | None = None,
    ) -> None:
        self.data_connector = data_connector
        self.chunksize = chunksize
        self.cache_manager = CacherManager()
        self.identity = identity or self.data_connector.identity or str(id(self))

        if not cacher_kwargs:
            cacher_kwargs = {}
        cacher_kwargs.setdefault("blocksize", self.chunksize)
        cacher_kwargs.setdefault("identity", self.data_connector.identity)
        if isinstance(cacher, Cacher):
            self.cacher = cacher
        elif isinstance(cacher, str) or isinstance(cacher, type):
            self.cacher = self.cache_manager.init_cacher(cacher, **cacher_kwargs)
        else:
            self.cacher = self.cache_manager.init_cacher(self.DEFAULT_CACHER, **cacher_kwargs)

        self.cacher.clear_invalid_cache()

        if isinstance(data_connector, GeneratorConnector):
            if isinstance(self.cacher, NoCache):
                raise DataLoaderInitError("NoCache can't be used with GeneratorConnector")
            # Warmup cache for generator, this allows random access
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

    def __getitem__(self, key: int | slice | list) -> pd.DataFrame:
        """
        Support get data by index and slice.
        """
        if isinstance(key, int):
            return self.cacher.load(
                offset=(key // self.chunksize) * self.chunksize,
                chunksize=self.chunksize,
                data_connector=self.data_connector,
            )[0]

        if isinstance(key, list):
            return pd.concat((d[key] for d in self.iter()), ignore_index=True)

        assert isinstance(key, slice)
        start = key.start or 0
        stop = key.stop or len(self)
        step = key.step or 1

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
