from __future__ import annotations

from typing import Any, Dict, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class NumericValueTransformer(Transformer):
    """
    Transformer class for handling numeric value (int + float) in data.
    """

    standard_scale: bool = True

    int_columns: Set = []

    float_columns: Set = []

    scalers: Dict = {}

    def fit(
        self,
        metadata: Metadata | None = None,
        tabular_data: DataLoader | pd.DataFrame = None,
        **kwargs: dict[str, Any],
    ):
        """
        The fit method.

        Data columns of int and float types need to be recorded here (Get data from metadata).
        """

        # TODO The methods to obtain these data types need to be changed
        self.int_columns = metadata.int_columns
        self.float_columns = metadata.float_columns

        if len(self.int_columns) == 0 and len(self.float_columns) == 0:
            logger.info("NumericValueTransformer Fitted (No numeric columns).")
            return

        # fit each columnxf
        for each_col in list(self.int_columns) + list(self.float_columns):
            self._fit_column(each_col, tabular_data[[each_col]])

        self.fitted = True
        logger.info("NumericValueTransformer Fitted.")

    def _fit_column(self, column_name: str, column_data: pd.DataFrame) -> np.ndarray:
        """
        Fit every numeric (include int and float) column in `_fit_column`.
        """

        if self.standard_scale:
            self._fit_column_scale(column_name, column_data)
            return

        return

    def _fit_column_scale(self, column_name: str, column_data: pd.DataFrame) -> np.ndarray:
        """
        Fit every numeric (include int and float) column using sklearn StandardScaler.
        """

        self.scalers[column_name] = StandardScaler()
        self.scalers[column_name].fit(column_data)

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle missing values in the input data.
        """

        logger.info("Converting data using NumericValueTransformer...")

        if len(self.int_columns) == 0 and len(self.float_columns) == 0:
            logger.info("Converting data using NumericValueTransformer... Finished (No column).")
            return

        processed_data = raw_data.copy()

        for each_col in list(self.int_columns) + list(self.float_columns):
            # convert every column then change the column
            processed_col = self._covert_column(each_col, processed_data[[each_col]])
            processed_data[each_col] = processed_col

        logger.info("Converting data using NumericValueTransformer... Finished.")
        return processed_data

    def _covert_column(self, column_name: str, column_data: pd.DataFrame):
        """
        Convert every numeric (include int and float) column.
        """

        if self.standard_scale:
            return self._covert_column_scale(column_name=column_name, column_data=column_data)
        pass

    def _covert_column_scale(self, column_name: str, column_data: pd.DataFrame):
        """
        Convert every numeric (include int and float) column using sklearn StandardScaler.
        """

        scaled_data = self.scalers[column_name].transform(column_data)
        return scaled_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse convert method, convert generated data into processed data.
        """

        for each_col in list(self.int_columns) + list(self.float_columns):
            # reverse convert every column then change the column
            processed_col = self._reverse_convert_column(each_col, processed_data[[each_col]])
            processed_data[each_col] = processed_col

        logger.info("Data reverse-converted by NumericValueTransformer (No Action).")

        return processed_data

    def _reverse_convert_column(self, column_name: str, column_data: pd.DataFrame):
        """
        Reverse convert method for each column.
        """

        if self.standard_scale:
            return self._reverse_convert_column_scale(
                column_name=column_name, column_data=column_data
            )
        return

    def _reverse_convert_column_scale(self, column_name: str, column_data: pd.DataFrame):
        """
        Reverse convert method for input column using scale method.
        """

        reverse_converted_data = self.scalers[column_name].inverse_transform(column_data)
        return reverse_converted_data

    pass


@hookimpl
def register(manager):
    manager.register("NumericValueTransformer", NumericValueTransformer)
