from __future__ import annotations

from typing import Generator

import pandas as pd

from sdgx.data_connectors.base import DataConnector


class CsvConnector(DataConnector):
    @property
    def identity(self):
        return self.path

    def __init__(
        self,
        path,
        sep=",",
        header="infer",
        **read_csv_kwargs,
    ):
        self.path = path
        self.sep = sep
        self.header = header
        self.read_csv_kwargs = read_csv_kwargs

    def _read(self, offset=0, limit=None) -> pd.DataFrame:
        return pd.read_csv(
            self.path,
            sep=self.sep,
            header=self.header,
            skiprows=offset,
            nrows=limit,
            **self.read_csv_kwargs,
        )

    def _columns(self) -> list[str]:
        d = pd.read_csv(
            self.path,
            sep=self.sep,
            header=self.header,
            nrows=0,
        ).columns.tolist()
        return d

    def iter(self, offset=0, chunksize=1000) -> Generator[pd.DataFrame, None, None]:
        for d in pd.read_csv(
            self.path,
            sep=self.sep,
            header=self.header,
            skiprows=offset,
            chunksize=chunksize,
            **self.read_csv_kwargs,
        ):
            yield d


from sdgx.data_connectors.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("CsvConnector", CsvConnector)
