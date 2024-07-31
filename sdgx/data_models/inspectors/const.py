from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class ConstInspector(Inspector):
    """
    ConstInspector is a class designed to identify columns in a DataFrame that contain constant values.
    It extends the base Inspector class and is used to fit the data and inspect it for constant columns.

    Attributes:
        const_columns (set[str]): A set of column names that contain constant values.
        const_values (dict[Any]): A dictionary mapping column names to their constant values.
        _inspect_level (int): The inspection level for this inspector, set to 80.
    """

    const_columns: set[str] = set()
    """
    A set of column names that contain constant values. This attribute is populated during the fit method by identifying columns in the DataFrame where all values are the same.
    """

    const_values: dict[Any] = {}
    """
    A dictionary mapping column names to their constant values. This attribute is populated during the fit method by storing the unique value found in each constant column.
    """

    _inspect_level = 80
    """
    The inspection level for this inspector, set to 80. This attribute indicates the priority or depth of inspection that this inspector performs relative to other inspectors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """
        Fit the inspector to the raw data.

        This method identifies columns in the DataFrame that contain constant values. It populates the `const_columns` set with the names of these columns and the `const_values` dictionary with the constant values found in each column.

        Args:
            raw_data (pd.DataFrame): The raw data to be inspected.

        Returns:
            None
        """
        self.const_columns = set()
        # iterate each column
        for column in raw_data.columns:
            if len(raw_data[column].value_counts(normalize=True)) == 1:
                self.const_columns.add(column)
                # self.const_values[column] = raw_data[column][0]

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"const_columns": self.const_columns}


@hookimpl
def register(manager):
    manager.register("ConstInspector", ConstInspector)
