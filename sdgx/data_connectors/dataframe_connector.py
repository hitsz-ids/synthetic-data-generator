from __future__ import annotations

import os
from functools import cached_property
from typing import Callable, Generator

import pandas as pd

from sdgx.data_connectors.base import DataConnector


class DataFrameConnector(DataConnector):
    def __init__(
        self,
        df: pd.DataFrame,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.df: pd.DataFrame = df

    def _read(self, offset: int = 0, limit: int | None = None) -> pd.DataFrame | None:
        length = self.df.shape[0]
        if offset >= length:
            return None
        limit = limit or length
        return self.df.iloc[offset : min(offset + limit, length)]

    def _columns(self) -> list[str]:
        return list(self.df.columns)

    def _iter(self, offset=0, chunksize=0) -> Generator[pd.DataFrame, None, None]:
        def generator() -> Generator[pd.DataFrame, None, None]:
            length = self.df.shape[0]
            if offset < length:
                current = offset
                while current < length:
                    yield self.df.iloc[current : min(current + chunksize, length - 1)]
                    current += chunksize

        return generator()

    def reset_dataframe(self, df: pd.DataFrame):
        self.df = df
