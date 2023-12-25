from __future__ import annotations

import hashlib
from functools import cached_property
from typing import Generator

import pandas as pd

from sdgx.data_connectors.base import DataConnector


class CsvConnector(DataConnector):
    """
    Wraps csv file into :ref:`DataConnector`

    Args:
        path (str): Path to csv file
        sep (str, optional): Separator. Defaults to ','.
        header (str, optional): Header. Defaults to 'infer'.
        read_csv_kwargs (dict, optional): kwargs for pd.read_csv, please refer to https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    Example:

        .. code-block:: python

            from sdgx.data_connectors.csv_connector import CsvConnector
            connector = CsvConnector(
                path="data.csv",
            )
            df = connector.read()


    """

    @cached_property
    def identity(self):
        """
        Identity of the data source is the sha256 of the file
        """
        with open(self.path, "rb") as f:
            return f"csvfile-{hashlib.sha256(f.read()).hexdigest()}"

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

    def _read(self, offset: int = 0, limit: int | None = None) -> pd.DataFrame | None:
        return pd.read_csv(
            self.path,
            sep=self.sep,
            header=self.header,
            skiprows=range(1, offset + 1),  # don't skip header
            nrows=limit,
            **self.read_csv_kwargs,
        )

    def _columns(self) -> list[str]:
        d = pd.read_csv(
            self.path,
            sep=self.sep,
            header=self.header,
            nrows=0,
            **self.read_csv_kwargs,
        ).columns.tolist()
        return d

    def _iter(self, offset: int = 0, chunksize: int = 1000) -> Generator[pd.DataFrame, None, None]:
        if chunksize is None:
            yield self._read(offset=offset)
            return

        for d in pd.read_csv(
            self.path,
            sep=self.sep,
            header=self.header,
            skiprows=range(1, offset + 1),  # don't skip header
            chunksize=chunksize,
            **self.read_csv_kwargs,
        ):
            yield d


from sdgx.data_connectors.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("CsvConnector", CsvConnector)
