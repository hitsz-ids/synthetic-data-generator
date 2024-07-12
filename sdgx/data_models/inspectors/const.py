from __future__ import annotations

from typing import Any
import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class ConstInspector(Inspector):

    const_columns: set[str] = set()

    const_values : dict[Any] = {}
    
    _inspect_level = 80
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of const columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """

        # Identify columns where the const rate exceeds the threshold
        self.const_columns = set()

        # iterate each column 
        for column in raw_data.columns:
            if len(raw_data[column].value_counts(normalize=True)) == 1:
                self.const_columns.add(column)
                self.const_values[column] = raw_data[column][0]

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"const_columns": list(self.const_columns),
                "const_values": self.const_values}


@hookimpl
def register(manager):
    manager.register("ConstInspector", ConstInspector)
