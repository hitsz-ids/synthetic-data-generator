from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Generator

import pandas as pd

from sdgx.data_exporters.base import DataExporter
from sdgx.exceptions import CannotExportError


class CsvExporter(DataExporter):
    def __init__(self, **kwargs):
        self.to_csv_kwargs = kwargs
        if "header" in self.to_csv_kwargs:
            self.to_csv_kwargs.pop("header")
        if "index" in self.to_csv_kwargs:
            self.to_csv_kwargs.pop("index")

    def write(
        self,
        dst: str | Path,
        data: pd.DataFrame | Generator[pd.DataFrame, None, None],
    ) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_csv(dst, index=False, **self.to_csv_kwargs)
        elif isinstance(data, Generator):
            with open(dst, "a") as file:
                for df in data:
                    df.to_csv(file, header=file.tell() == 0, index=False, **self.to_csv_kwargs)
        else:
            raise CannotExportError(f"Cannot export data of type {type(data)} to csv")


from sdgx.data_exporters.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("CsvExporter", CsvExporter)
