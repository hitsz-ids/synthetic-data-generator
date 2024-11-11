from __future__ import annotations

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

    fixed_combinations: dict[str, set[str]]
    """
    A dictionary mapping column names to sets of column names that have fixed relationships with them.
    """

    def __init__(self):
        super().__init__()  # Call the parent class's initialization method
        self.fixed_combinations: dict[str, set[str]] = (
            {}
        )  # Initialize the variable in the initialization method
        self.column_ratios = {}  # New: Save the ratio relationships between columns

    def fit(self, metadata: Metadata, **kwargs):
        """Fit the transformer and save the relationships between columns.

        Args:
            metadata (Metadata): Metadata object
        """
        self.fixed_combinations = metadata.get("fixed_combinations")
        self.column_ratios = {}  # New: Save the ratio relationships between columns

        # Calculate and save the ratio relationships between columns from the raw data
        if "raw_data" in kwargs:
            raw_data = kwargs["raw_data"]
            for base_col, related_cols in self.fixed_combinations.items():
                base_values = raw_data[base_col]
                for related_col in related_cols:
                    if related_col in raw_data.columns:
                        # Calculate the ratio relationship (assuming a linear relationship)
                        ratio = (raw_data[related_col] / base_values).mean()
                        self.column_ratios[(base_col, related_col)] = ratio

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method to handle columns with fixed combinations in the input data.

        保留所有列的数据，不删除任何列，以防止后续组件报错。

        Args:
            raw_data (pd.DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: The original DataFrame without any modifications.
        """
        processed_data = raw_data.copy()

        logger.info("Converting data using FixedCombinationTransformer... 保留所有列的数据。")

        # If additional processing is needed during the conversion, it can be added here.
        # For example, a marker column can be added to indicate which columns have fixed combination relationships.

        logger.info("Converting data using FixedCombinationTransformer... Finished.")

        return processed_data

    def reverse_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse convert data by replacing the values of fixed combination columns using saved ratio relationships.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with the values of fixed combination columns replaced based on the saved ratios
        """
        result_df = df.copy()

        logger.info("Reverse converting data using FixedCombinationTransformer...")

        for base_col, related_cols in self.fixed_combinations.items():
            if base_col in df.columns:
                base_data = df[base_col]
                for related_col in related_cols:
                    if related_col in df.columns:
                       
                        ratio = self.column_ratios.get((base_col, related_col), 2)  # 默认比例为2
                        original_values = base_data * ratio
                        result_df[related_col] = original_values
                        logger.debug(f"Replaced values in column {related_col} using ratio {ratio} based on {base_col}.")

        logger.info("Reverse converting data using FixedCombinationTransformer... Finished.")

        return result_df

@hookimpl
def register(manager):
    manager.register("FixedCombinationTransformer", FixedCombinationTransformer)
