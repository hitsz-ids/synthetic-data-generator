from __future__ import annotations

import os
from functools import cached_property
from typing import Callable, Generator

import pandas as pd

from sdgx.data_connectors.base import DataConnector


class GeneratorConnector(DataConnector):
    """
    A virtual data connector that wrap
    `Generator <https://docs.python.org/3/glossary.html#term-generator>`_
    into a DataConnector.

    Passing ``offset=0`` to ``read`` will reset the generator.

    Warning:
        ``offset`` and ``limit`` are ignored as ``Generator`` not supporting random access.
        But we can use :ref:`Cacher` to support it. See :ref:`Data Loader` for more details.

    Note:
        This connector is not been registered by default.
        So only be used with the library way.
    """

    @cached_property
    def identity(self) -> str:
        return f"generator-{os.getpid()}-{id(self.generator_caller)}"

    def __init__(
        self,
        generator_caller: Callable[[], Generator[pd.DataFrame, None, None]],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.generator_caller = generator_caller
        self._generator = self.generator_caller()

    def _read(self, offset: int = 0, limit: int | None = None) -> pd.DataFrame | None:
        """
        Ingore limit and allow sequential reading.
        """
        if offset == 0:
            self._generator = self.generator_caller()

        try:
            return next(self._generator)
        except StopIteration:
            return None

    def _columns(self) -> list[str]:
        for df in self._iter():
            return list(df.columns)

    def _iter(self, offset=0, chunksize=0) -> Generator[pd.DataFrame, None, None]:
        """
        Subclass should implement this for reading data in chunk.

        See ``iter`` for more details.
        """
        return self.generator_caller()
