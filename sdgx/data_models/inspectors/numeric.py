from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class NumericInspector(Inspector):
    """
    A class for inspecting numeric data.

    This class is a subclass of `Inspector` and is designed to provide methods for inspecting
    and analyzing numeric data. It includes methods for detecting int or float data type.

    In August 2024, we introduced a new feature that will continue to judge the positivity or
    negativity after determining the type, thereby effectively improving the quality of synthetic
    data in subsequent processing.
    """

    int_columns: set = set()
    """
    A set of column names that contain integer values.
    """

    float_columns: set = set()
    """
    A set of column names that contain float values.
    """

    positive_columns: set = set()
    """
    A set of column names that contain only positive numeric values.
    """

    negative_columns: set = set()
    """
    A set of column names that contain only negative numeric values.
    """

    pos_threshold: float = 0.95
    """
    The threshold proportion of positive values in a column to consider it as a positive column.
    """

    negative_threshold: float = 0.95
    """
    The threshold proportion of negative values in a column to consider it as a negative column.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._int_rate = 0.9
        self.df_length = 0

    def _is_int_column(self, col_series: pd.Series) -> bool:
        """
        Determine if a column contains predominantly integer values.

        This method checks if the proportion of integer values in the given column
        exceeds a predefined threshold.

        Args:
            col_series (pd.Series): The column series to be inspected.

        Returns:
            bool: True if the column is predominantly integer, False otherwise.
        """
        # Convert the column series to numeric values, coercing errors to NaN and dropping them
        numeric_values = pd.to_numeric(col_series, errors="coerce").dropna()

        # If there are no numeric values, return False to avoid division by zero
        if len(numeric_values) == 0:
            return False

        # Count how many of the numeric values are integers
        int_cnt = (numeric_values == numeric_values.astype(int)).sum()

        # Calculate the ratio of integer values to the total numeric values
        int_rate = int_cnt / len(numeric_values)

        # Return True if the integer rate is greater than the predefined threshold
        return int_rate > self._int_rate

    def _is_positive_or_negative_column(
        self, col_series: pd.Series, threshold: float, comparison_func
    ) -> bool:
        """
        Determine if a column contains predominantly positive or negative values.

        This method checks if the proportion of values that satisfy a given comparison
        function exceeds a predefined threshold.

        Args:
            col_series (pd.Series): The column series to be inspected.
            threshold (float): The proportion threshold for considering the column as positive or negative.
            comparison_func (function): A function that takes a numeric value and returns a boolean.

        Returns:
            bool: True if the column satisfies the condition, False otherwise.
        """
        # Convert the column series to numeric values, coercing errors to NaN and dropping NaN values
        numeric_values = pd.to_numeric(col_series, errors="coerce").dropna()

        # If there are no numeric values, return False to avoid division by zero
        if len(numeric_values) == 0:
            return False

        # Apply the comparison function to the numeric values and sum the results
        count = comparison_func(numeric_values).sum()

        # Calculate the proportion of values that meet the comparison criteria
        proportion = count / len(numeric_values)

        # Return True if the proportion meets or exceeds the threshold, otherwise False
        return proportion >= threshold

    def _is_positive_column(self, col_series: pd.Series) -> bool:
        """
        Determine if a column contains predominantly positive values.

        This method checks if the proportion of positive values in the given column
        exceeds a predefined threshold.

        Args:
            col_series (pd.Series): The column series to be inspected.

        Returns:
            bool: True if the column is predominantly positive, False otherwise.
        """
        return self._is_positive_or_negative_column(col_series, self.pos_threshold, lambda x: x > 0)

    def _is_negative_column(self, col_series: pd.Series) -> bool:
        """
        Determine if a column contains predominantly negative values.

        This method checks if the proportion of negative values in the given column
        exceeds a predefined threshold.

        Args:
            col_series (pd.Series): The column series to be inspected.

        Returns:
            bool: True if the column is predominantly negative, False otherwise.
        """
        return self._is_positive_or_negative_column(
            col_series, self.negative_threshold, lambda x: x < 0
        )

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """

        # Initialize sets for integer and float columns
        self.int_columns = set()
        self.float_columns = set()

        # Initialize sets for positive and negative columns
        self.positive_columns = set()
        self.negative_columns = set()

        # Store the length of the DataFrame
        self.df_length = len(raw_data)

        # Iterate all columns and determain the final data type
        for col in raw_data.columns:
            if raw_data[col].dtype in ["int64", "float64"]:
                # float or int
                if self._is_int_column(raw_data[col]):
                    self.int_columns.add(col)
                else:
                    self.float_columns.add(col)

                # positive? negative?
                if self._is_positive_column(raw_data[col]):
                    self.positive_columns.add(col)
                elif self._is_negative_column(raw_data[col]):
                    self.negative_columns.add(col)

        # Mark the inspector as ready
        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        # Positive and negative columns should not be strictly considered as label columns
        # We use the format dict to inspect and output to metadata
        numeric_format: dict = {}
        numeric_format["positive"] = sorted(list(self.positive_columns))
        numeric_format["negative"] = sorted(list(self.negative_columns))

        return {
            "int_columns": list(self.int_columns),
            "float_columns": list(self.float_columns),
            "numeric_format": numeric_format,
        }


@hookimpl
def register(manager):
    manager.register("NumericInspector", NumericInspector)
