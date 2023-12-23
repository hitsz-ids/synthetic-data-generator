from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class NumericInspector(Inspector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numeric_columns: set[str] = set()

    def fit(self, raw_data: pd.DataFrame):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """

        self.numeric_columns = self.numeric_columns.union(
            set(raw_data.select_dtypes(include=['float64', "int64"]).columns)
        )
        self.ready = True

    def inspect(self) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"numeric_columns": list(self.numeric_columns)}


@hookimpl
def register(manager):
    manager.register("NumericInspector", NumericInspector)
