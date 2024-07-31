from __future__ import annotations

from typing import Any

import pandas as pd
from pandas._libs.tslibs.parsing import DateParseError

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class BoolInspector(Inspector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bool_columns: set[str] = set()

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        self.bool_columns = set()
        self.bool_columns = self.bool_columns.union(
            set(raw_data.infer_objects().select_dtypes(include=["bool"]).columns)
        )

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"bool_columns": list(self.bool_columns)}


@hookimpl
def register(manager):
    manager.register("BoolInspector", BoolInspector)
