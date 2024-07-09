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

    empty_columns: list = []
    """
    List of column names that are identified as empty. This attribute is populated during the fitting process
    and is used to remove these columns during the conversion process and restore them during the reverse conversion process.
    """


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

        self.empty_columns = list(metadata.get('empty_columns'))

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
            processed_data = self.remove_columns(raw_data, each_col)
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

        df_length = processed_data.shape[0]

        for each_col_name in self.empty_columns:
            each_empty_col = [None for _ in range(df_length)]
            each_empty_df = pd.DataFrame({each_col_name: each_empty_col})
            processed_data = self.attach_columns(processed_data, each_empty_df)

        logger.info("Data reverse-converted by EmptyTransformer.")

        return processed_data

@hookimpl
def register(manager):
    manager.register("EmptyTransformer", EmptyTransformer)
