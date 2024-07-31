from __future__ import annotations

from typing import Any

import pandas as pd
from pandas._libs.tslibs.parsing import DateParseError

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.utils import ignore_warnings


class DatetimeInspector(Inspector):
    _inspect_level = 20
    """
    The inspect_level of DatetimeInspector is higher than DiscreteInspector.

    Often, difficult-to-recognize date or datetime objects are also recognized as descrete types by DatetimeInspector, causing the column to be marked repeatedly.
    """

    _format_match_rate = 0.9
    """
    When specifically check the datatime format, problems caused by missing values and incorrect values will inevitably occur.
    To fix this, we discard the .any()  method and use the `match_rate` to increase the robustness of this inspector.
    """

    PRESET_FORMAT_STRINGS = [
        "%Y-%m-%d",
        "%d %b %Y",
        "%b-%Y",
        "%Y/%m/%d",
    ]

    def __init__(self, user_formats: list[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datetime_columns: set[str] = set()
        self.user_defined_formats = user_formats if user_formats else []
        self.column_formats: dict[str, str] = {}

    @classmethod
    @ignore_warnings(category=UserWarning)
    def can_convert_to_datetime(cls, input_col: pd.Series):
        """Whether a df column can be converted to datetime.

        Args:
            input_col(pd.Series): A column of a dataframe.
        """
        try:
            pd.to_datetime(input_col)
            return True
        except DateParseError:
            return False
        # for other situations
        except:
            return False

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Gets the list of discrete columns from the raw data.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        self.datetime_columns = set()

        self.datetime_columns = self.datetime_columns.union(
            set(raw_data.infer_objects().select_dtypes(include=["datetime64"]).columns)
        )

        # for some other case
        # Some columns containing dates after infer are still marked as object
        candidate_columns = set(raw_data.select_dtypes(include=["object"]).columns)
        for col_name in candidate_columns:
            each_col = raw_data[col_name]
            if DatetimeInspector.can_convert_to_datetime(each_col):
                self.datetime_columns.add(col_name)

        # Process for detecting format strings
        for col_name in self.datetime_columns:
            each_col = raw_data[col_name]
            datetime_format = self.detect_datetime_format(each_col)
            if datetime_format:
                self.column_formats[col_name] = datetime_format

        self.ready = True

    def detect_datetime_format(self, series: pd.Series):
        """Detects the datetime format of a pandas Series.

        This method iterates over a list of user-defined and preset datetime formats,
        and attempts to parse each date in the series using each format.
        If all dates in the series can be successfully parsed with a format,
        that format is returned. If no format can parse all dates, an empty string is returned.

        Args:
            series (pd.Series): The pandas Series to detect the datetime format of.

        Returns:
               str: The datetime format that can parse all dates in the series, or None if no such format is found.
        """

        def _is_series_fit_format(parsed_series, match_rate):
            length = len(parsed_series)
            false_num = len(list(i for i in parsed_series if i is False))
            false_rate = false_num / length
            return false_rate >= match_rate

        for fmt in self.user_defined_formats + self.PRESET_FORMAT_STRINGS:
            try:
                # Check if all dates in the series can be parsed with this format
                parsed_series = series.apply(
                    lambda x: pd.to_datetime(x, format=fmt, errors="coerce")
                )
                # if fit return format, return
                if _is_series_fit_format(parsed_series.isnull(), self._format_match_rate):
                    return fmt
            except ValueError:
                continue

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {
            "datetime_columns": list(self.datetime_columns),
            "datetime_formats": self.column_formats,
        }


@hookimpl
def register(manager):
    manager.register("DatetimeInspector", DatetimeInspector)
