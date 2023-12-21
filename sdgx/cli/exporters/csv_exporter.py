from pathlib import Path
from typing import Generator

import pandas as pd

from sdgx.cli.exporters.base import Exporter
from sdgx.exceptions import CannotExportError


class CsvExporter(Exporter):
    def write(
        self,
        dst: str | Path,
        data: pd.DataFrame | Generator[pd.DataFrame, None, None],
    ) -> None:
        if isinstance(data, pd.DataFrame):
            data.to_csv(dst, index=False)
        elif isinstance(data, Generator):
            with open(dst, "a") as file:
                for df in data:
                    df.to_csv(file, header=file.tell() == 0, index=False)
        else:
            raise CannotExportError(f"Cannot export data of type {type(data)} to csv")


from sdgx.cli.exporters.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("CsvExporter", CsvExporter)
