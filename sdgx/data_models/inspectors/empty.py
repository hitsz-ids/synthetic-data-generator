from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class EmptyInspector(Inspector):
    """
    The EmptyInspector class is designed to identify columns in a DataFrame that have a high rate of missing values.

    Columns taged empty will be removed during the training process and reinserted into their original positions after the model sampling process is complete.

    Attributes:
        empty_rate_threshold (float): The threshold for the rate of missing values above which a column is considered empty, default = 0.9.
        empty_columns (set[str]): A set of column names that have missing values above the threshold.

    Methods:
        __init__(self, *args, **kwargs): Initializes the EmptyInspector instance, optionally setting the empty_rate_threshold.
        fit(self, raw_data: pd.DataFrame, *args, **kwargs): Fits the inspector to the raw data, identifying columns with missing values above the threshold.
        inspect(self, *args, **kwargs) -> dict[str, Any]: Returns a dictionary containing the list of columns identified as empty.
    """

    empty_rate_threshold = 0.9
    """
    float: The threshold for the rate of missing values above which a column is considered empty.
    Default is 0.9, meaning if a column has more than 90% of its values missing, it will be considered empty.
    """

    empty_columns: set[str] = set()
    """
    set[str]: A set of column names that have missing values above the empty_rate_threshold.
    These columns are identified as empty and will be handled accordingly during the data processing.
    """

    _inspect_level = 90
    """
    int: The inspection level for the EmptyInspector, set to a quite high value (90) to prioritize the identification and handling of empty columns.
    This high value is chosen because empty columns contain no information and should not be considered for any other type of inspection or processing.
    They are typically removed during model training as they cannot be understood by many models and may cause errors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "empty_rate_threshold" in kwargs:
            self.empty_rate_threshold = kwargs["empty_rate_threshold"]

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of empty columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        # Calculate the empty rate for each column
        empty_rate = raw_data.isnull().mean()

        # Identify columns where the empty rate exceeds the threshold
        self.empty_columns = set(empty_rate[empty_rate >= self.empty_rate_threshold].index)

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"empty_columns": list(self.empty_columns)}


@hookimpl
def register(manager):
    manager.register("EmptyInspector", EmptyInspector)
