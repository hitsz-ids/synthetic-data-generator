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

    fixed_combinations: dict[str, set[str]] = {}
    """
    A dictionary mapping column names to sets of column names that have fixed relationships with them.
    """

    def fit(self, metadata: Metadata, **kwargs):
        """Fit the transformer and save the relationships between columns.
    
        Args:
            metadata (Metadata): Metadata object
        """
        self.fixed_combinations = metadata.get("fixed_combinations", {})
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

    def reverse_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse convert data, reconstructing the removed fixed combination columns using saved ratio relationships.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame containing the reconstructed columns
        """
        result_df = df.copy()
        
        for base_col, related_cols in self.fixed_combinations.items():
            if base_col in df.columns:
                base_data = df[base_col]
                for related_col in related_cols:
                    if related_col not in df.columns:
                        # Reconstruct the column using the saved ratio relationship
                        ratio = self.column_ratios.get((base_col, related_col), 2)  # Default value is 2
                        result_df[related_col] = base_data * ratio
        
        return result_df


@hookimpl
def register(manager):
    manager.register("FixedCombinationTransformer", FixedCombinationTransformer)
