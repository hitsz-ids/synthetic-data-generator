from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class ColumnOrderTransformer(Transformer):
    """
    A transformer that rearranges the columns of a DataFrame to a specified order.

    Attributes:
        column_list (list): The list of column names in the desired order.

    Methods:
        fit(metadata: Metadata | None = None, **kwargs: dict[str, Any]): Fits the transformer by remembering the order of the columns.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame: Converts the input DataFrame by rearranging its columns.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame: Reverse-converts the processed DataFrame by rearranging its columns back to their original order.
        rearrange_columns(column_list, processed_data): Rearranges the columns of a DataFrame according to the provided column list.
    """

    column_list: list
    """
    The list of tabular data's columns.
    """

    def __init__(self):
        self.column_list = None

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        Remember the order of the columns.
        """

        self.column_list = list(metadata.column_list)

        logger.info("ColumnOrderTransformer Fitted.")

        self.fitted = True

        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle missing values in the input data.
        """
        logger.info("Converting data using ColumnOrderTransformer...")
        logger.info("Converting data using ColumnOrderTransformer... Finished (No action).")

        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.
        """

        res = self.rearrange_columns(self.column_list, processed_data)
        logger.info("Data reverse-converted by ColumnOrderTransformer.")

        return res

    @staticmethod
    def rearrange_columns(column_list, processed_data):
        """
        This method rearranges the columns of a given DataFrame according to the provided column list.

        Any columns in the DataFrame that are not in the column list are dropped.

        Args:
            - column_list (list): A list of column names in the order they should appear in the output DataFrame.
            - processed_data (pd.DataFrame): The DataFrame to be rearranged.

        Returns:
            - result_data (pd.DataFrame): The rearranged DataFrame.
        """
        result_data = processed_data.reindex(columns=column_list)

        return result_data


@hookimpl
def register(manager):
    manager.register("ColumnOrderTransformer", ColumnOrderTransformer)
