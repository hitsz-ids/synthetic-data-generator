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
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.int_columns: set[str] = set()
        self.float_columns: set[str] = set()
        self._int_rate = 0.9
        self.df_length = 0

    def is_int_column(self, col_series: pd.Series):
        """
        Determine whether a column of pd.DataFrame is of type int
        In the original pd.DataFrame automatically updated dtype, some int types will be marked as float.
        In fact, we can make an accurate result by getting the decimal part of the value.

        Args:
            col_series (pd.Series): One single column of the raw data.
        """

        def is_decimal_part_zero(num: float):
            """
            Is the decimal part == 0.0 ?

            Args:
                col_series (float): The number.
            """
            try:
                decimal_part = num - int(num)
            except ValueError:
                return None
            if decimal_part == 0.0:
                return True
            else:
                return False
        # Initialize the counter for values with zero decimal part
        int_cnt = 0
        col_length = self.df_length

        # Iterate over each value in the series
        for each_val in col_series:
            decimal_zer0 = is_decimal_part_zero(each_val)
            # If the decimal part is zero, increment the counter and continue to the next value
            if decimal_zer0 is True:
                int_cnt += 1
                continue
            # If the decimal part is not zero or not a decimal number
            # decrease the length of the series and continue to the next value
            if decimal_zer0 is None:
                col_length -= 1
                continue
        
        # Calculate the rate of values with zero decimal part
        if col_length <= 0:
            int_rate = 0
        else:
            int_rate = int_cnt / col_length
        
        # Check if the rate is greater than the predefined rate
        if int_rate > self._int_rate:
            return True
        else:
            return False

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """

        self.df_length = len(raw_data)

        float_candidate = self.float_columns.union(
            set(raw_data.select_dtypes(include=["float64"]).columns)
        )

        for candidate in float_candidate:
            if self.is_int_column(raw_data[candidate]):
                self.int_columns.add(candidate)
            else:
                self.float_columns.add(candidate)

        self.int_columns = self.int_columns.union(
            set(raw_data.select_dtypes(include=["int64"]).columns)
        )

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {
            "int_columns": list(self.int_columns),
            "float_columns": list(self.float_columns),
        }

@hookimpl
def register(manager):
    manager.register("NumericInspector", NumericInspector)
