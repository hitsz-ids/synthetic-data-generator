from __future__ import annotations

from typing import Any

import pandas as pd
from pandas._libs.tslibs.parsing import DateParseError

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.utils import ignore_warnings


class DatetimeInspector(Inspector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datetime_columns: set[str] = set()

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

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"datetime_columns": list(self.datetime_columns)}


@hookimpl
def register(manager):
    manager.register("DatetimeInspector", DatetimeInspector)
