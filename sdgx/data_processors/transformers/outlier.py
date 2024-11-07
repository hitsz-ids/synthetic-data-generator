from __future__ import annotations

from typing import Any

from pandas import DataFrame

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class OutlierTransformer(Transformer):
    """
    A transformer class to handle outliers in the data by converting them to specified fill values.

    Attributes:
        int_columns (set): A set of column names that contain integer values.
        int_outlier_fill_value (int): The value to fill in for outliers in integer columns. Default is 0.
        float_columns (set): A set of column names that contain float values.
        float_outlier_fill_value (float): The value to fill in for outliers in float columns. Default is 0.
    """

    int_columns: set
    """
    set: A set of column names that contain integer values. These columns will have their outliers replaced by `int_outlier_fill_value`.
    """

    int_outlier_fill_value: int
    """
    int: The value to fill in for outliers in integer columns. Default is 0.
    """

    float_columns: set
    """
    set: A set of column names that contain float values. These columns will have their outliers replaced by `float_outlier_fill_value`.
    """

    float_outlier_fill_value: float
    """
    float: The value to fill in for outliers in float columns. Default is 0.
    """

    def __init__(self):
        self.int_columns = set()
        self.int_outlier_fill_value = 0
        self.float_columns = set()
        self.float_outlier_fill_value = float(0)

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        Records the names of integer and float columns from the metadata.

        Args:
            metadata (Metadata | None): The metadata object containing column type information.
            **kwargs: Additional keyword arguments.
        """
        # int columns
        for each_col in metadata.int_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == "int":
                self.int_columns.add(each_col)

        # float columns
        for each_col in metadata.float_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == "float":
                self.float_columns.add(each_col)

        self.fitted = True

        logger.info("OutlierTransformer Fitted.")

    def convert(self, raw_data: DataFrame) -> DataFrame:
        """
        Convert method to handle outliers in the input data by replacing them with specified fill values.

        Args:
            raw_data (DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            DataFrame: The processed DataFrame with outliers replaced by fill values.
        """
        res = raw_data

        logger.info("Converting data using OutlierTransformer...")

        # Dealing with the integer value columns
        def convert_to_int(value):
            try:
                return int(value)
            except ValueError:
                return self.int_outlier_fill_value

        for each_col in self.int_columns:
            res[each_col] = res[each_col].apply(convert_to_int)

        # Dealing with the float value columns
        def convert_to_float(value):
            try:
                return float(value)
            except ValueError:
                return self.float_outlier_fill_value

        for each_col in self.float_columns:
            res[each_col] = res[each_col].apply(convert_to_float)

        logger.info("Converting data using OutlierTransformer... Finished.")

        return res

    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        """
        Reverse_convert method for the transformer (No action for OutlierTransformer).

        Args:
            processed_data (DataFrame): The processed DataFrame.

        Returns:
            DataFrame: The same processed DataFrame.
        """
        logger.info("Data reverse-converted by OutlierTransformer (No Action).")

        return processed_data


@hookimpl
def register(manager):
    """
    Register the OutlierTransformer with the manager.

    Args:
        manager: The manager object responsible for registering transformers.
    """
    manager.register("OutlierTransformer", OutlierTransformer)
