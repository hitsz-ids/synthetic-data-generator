from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class DiscreteInspector(Inspector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discrete_columns: set[str] = set()

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        self.discrete_columns = set()

        self.discrete_columns = self.discrete_columns.union(
            set(raw_data.select_dtypes(include="object").columns)
        )
        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"discrete_columns": list(self.discrete_columns)}


@hookimpl
def register(manager):
    manager.register("DiscreteInspector", DiscreteInspector)
