from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class IDInspector(Inspector):
    _inspect_level = 20
    """
    The inspect_level of IDInspector is higher than NumericInspector.

    Often, some column, especially int type id column can also be recognized as numeric types by NumericInspector, causing the column to be marked repeatedly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ID_columns: set[str] = set()

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """

        self.ID_columns = set()

        df_length = len(raw_data)
        candidate_columns = set(raw_data.select_dtypes(include=["object", "int64"]).columns)

        for each_col_name in candidate_columns:
            target_col = raw_data[each_col_name]
            col_set_length = len(set(target_col))
            if col_set_length == df_length:
                self.ID_columns.add(each_col_name)

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"id_columns": list(self.ID_columns)}


@hookimpl
def register(manager):
    manager.register("IDInspector", IDInspector)
