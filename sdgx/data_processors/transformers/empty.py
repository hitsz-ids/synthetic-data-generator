from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class EmptyTransformer(Transformer):
    """
    A transformer that handles empty columns in a DataFrame.

    This transformer identifies and processes columns that contain no data (empty columns) in a given DataFrame.
    It can remove these columns during the conversion process and restore them during the reverse conversion process.

    Attributes:
        empty_columns (list): A list of column names that are identified as empty.

    Methods:
        fit(metadata: Metadata | None = None, **kwargs: dict[str, Any]):
            Fits the transformer to the data by identifying empty columns based on provided metadata.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame:
            Converts the raw data by removing the identified empty columns.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame:
            Reverses the conversion by restoring the previously removed empty columns.
    """

    empty_columns: set
    """
    Set of column names that are identified as empty. This attribute is populated during the fitting process
    and is used to remove these columns during the conversion process and restore them during the reverse conversion process.
    """

    def __init__(self):
        self.empty_columns = set()

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.
        Remember the empty_columns from all columns.

        Args:
            metadata (Metadata | None): The metadata containing information about the data, including empty columns.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            None
        """
        for each_col in metadata.get("empty_columns"):
            if metadata.get_column_data_type(each_col) == "empty":
                self.empty_columns.add(each_col)

        logger.info("EmptyTransformer Fitted.")

        self.fitted = True

        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the raw data by removing the identified empty columns.

        Args:
            raw_data (pd.DataFrame): The input DataFrame containing the raw data.

        Returns:
            pd.DataFrame: The processed DataFrame with empty columns removed.
        """
        processed_data = raw_data

        logger.info("Converting data using EmptyTransformer...")

        for each_col in self.empty_columns:
            processed_data = self.remove_columns(processed_data, [each_col])
        logger.info("Converting data using EmptyTransformer... Finished (No action).")

        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the conversion by restoring the previously removed empty columns.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: The DataFrame with previously removed empty columns restored.
        """

        if not self.fitted or not self.empty_columns:
            return processed_data

        for col_name in self.empty_columns:
            # Create an empty column with the same number of rows as `processed_data`.
            empty_df = pd.DataFrame({col_name: [None] * len(processed_data)})
            processed_data = self.attach_columns(processed_data, empty_df)

        return processed_data


@hookimpl
def register(manager):
    manager.register("EmptyTransformer", EmptyTransformer)
