from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class ConstValueTransformer(Transformer):
    """
    A transformer that replaces the input with a constant value.

    This class is used to transform any input data into a predefined constant value.
    It is particularly useful in scenarios where a consistent output is required regardless of the input.

    Attributes:
        const_value (dict[Any]): The constant value that will be returned.
    """

    const_columns: list

    const_values: dict[Any, Any]

    def __init__(self):
        self.const_columns = []
        self.const_values = {}

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        This method processes the metadata to identify columns that should be replaced with a constant value.
        It updates the internal state of the transformer with the columns and their corresponding constant values.

        Args:
            metadata (Metadata | None): The metadata object containing information about the columns and their data types.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            None
        """

        for each_col in metadata.column_list:
            if metadata.get_column_data_type(each_col) == "const":
                self.const_columns.append(each_col)
                # self.const_values[each_col] = metadata.get("const_values")[each_col]

        logger.info("ConstValueTransformer Fitted.")

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle missing values in the input data by replacing specified columns with constant values.

        This method iterates over the columns identified for replacement with constant values and removes them from the input DataFrame.
        The removal is based on the columns specified during the fitting process.

        Args:
            raw_data (pd.DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns removed.
        """

        processed_data = copy.deepcopy(raw_data)

        logger.info("Converting data using ConstValueTransformer...")

        for each_col in self.const_columns:
            # record values here
            if each_col not in self.const_values.keys():
                self.const_values[each_col] = processed_data[each_col].unique()[0]
            processed_data = self.remove_columns(processed_data, [each_col])

        logger.info("Converting data using ConstValueTransformer... Finished.")

        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.

        This method restores the original columns that were replaced with constant values during the conversion process.
        It iterates over the columns identified for replacement with constant values and adds them back to the DataFrame
        with the predefined constant values.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: A DataFrame with the original columns restored, filled with their corresponding constant values.
        """
        df_length = processed_data.shape[0]

        for each_col_name in self.const_columns:
            each_value = self.const_values[each_col_name]
            each_const_col = [each_value for _ in range(df_length)]
            each_const_df = pd.DataFrame({each_col_name: each_const_col})
            processed_data = self.attach_columns(processed_data, each_const_df)

        logger.info("Data reverse-converted by ConstValueTransformer.")

        return processed_data


@hookimpl
def register(manager):
    manager.register("ConstValueTransformer", ConstValueTransformer)
