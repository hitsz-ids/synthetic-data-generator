from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class FixedCombinationTransformer(Transformer):
    """
    A transformer that handles columns with fixed combinations in a DataFrame.

    This transformer identifies and processes columns that have fixed relationships (high covariance) in a given DataFrame.
    It can remove these columns during the conversion process and restore them during the reverse conversion process.

    Attributes:
        fixed_combinations (dict[str, set[str]]): A dictionary mapping column names to sets of column names that have fixed relationships with them.
    """

    fixed_combinations: dict[str, set[str]] = {}
    """
    A dictionary mapping column names to sets of column names that have fixed relationships with them.
    """

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        This method processes the metadata to identify columns that have fixed relationships.
        It updates the internal state of the transformer with the columns and their corresponding fixed combinations.

        Args:
            metadata (Metadata | None): The metadata object containing information about the columns and their fixed combinations.
            **kwargs (dict[str, Any]): Additional keyword arguments.

        Returns:
            None
        """
        self.fixed_combinations = metadata.get("fixed_combinations")

        logger.info("FixedCombinationTransformer Fitted.")

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle columns with fixed combinations in the input data.

        This method iterates over the columns identified for fixed combinations and removes them from the input DataFrame.
        The removal is based on the columns specified during the fitting process.

        Args:
            raw_data (pd.DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns removed.
        """
        processed_data = raw_data.copy()

        logger.info("Converting data using FixedCombinationTransformer...")

        for column, related_columns in self.fixed_combinations.items():
            processed_data = self.remove_columns(processed_data, list(related_columns))

        logger.info("Converting data using FixedCombinationTransformer... Finished.")

        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the transformer.

        This method restores the original columns that were removed during the conversion process.
        It iterates over the columns identified for fixed combinations and adds them back to the DataFrame.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: A DataFrame with the original columns restored.
        """
        df_length = processed_data.shape[0]

        for _, related_columns in self.fixed_combinations.items():
            for related_column in related_columns:
                each_fixed_col = [None for _ in range(df_length)]
                each_fixed_df = pd.DataFrame({related_column: each_fixed_col})
                processed_data = self.attach_columns(processed_data, each_fixed_df)

        logger.info("Data reverse-converted by FixedCombinationTransformer.")

        return processed_data


@hookimpl
def register(manager):
    manager.register("FixedCombinationTransformer", FixedCombinationTransformer)
