from __future__ import annotations

import random

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

    def __init__(self):
        super().__init__()  # Call the parent class's initialization method
        # Initialize the variable in the initialization method

        self.fixed_combinations: dict[str, set[str]] = {}
        """
        A dictionary mapping column names to sets of column names that have fixed relationships with them.
        """

        self.simplified_fixed_combinations: dict[str, set[str]] = {}
        """
        A dictionary mapping column names to sets of column names that have fixed relationships with them.
        """

        self.column_mappings: dict[(str, str), dict[str, str]] = {}
        """
        A dictionary mapping tuples of column names to dictionaries of value mappings.
        """

    def fit(self, metadata: Metadata, **kwargs):
        """Fit the transformer and save the relationships between columns.

        Args:
            metadata (Metadata): Metadata object
        """
        self.fixed_combinations = metadata.get("fixed_combinations")

        # simplify the fixed_combinations, remove the symmetric and duplicate combinations
        simplified_fixed_combinations = {}
        seen = set()

        for base_col, related_cols in self.fixed_combinations.items():
            # create a immutable set of base_col and related_cols
            combination = frozenset([base_col]) | frozenset(related_cols)

            # if the combination has not been seen, add it to the simplified_fixed_combinations
            if combination not in seen:
                simplified_fixed_combinations[base_col] = related_cols
                seen.add(combination)

        self.simplified_fixed_combinations = simplified_fixed_combinations

        self.has_column_mappings = False

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert the input DataFrame by identifying and storing fixed column relationships.

        This method analyzes the relationships between columns specified in simplified_fixed_combinations
        and stores their value mappings. The mappings are only computed once for the first batch of data
        to optimize performance.

        NOTE:
            The current implementation has space for optimization:
            - If column_mappings already exist, no recalculation is performed
            - The column_mappings definition depends on the first batch of data from the DataLoader
            - This might miss some edge cases where column relationships are very comprehensive
              (e.g., some column correspondences might only appear in later batches)
            - While processing each batch separately could avoid this issue, it would incur
              significant performance overhead
            - The current function is sufficient for most scenarios
            - In the future, we may introduce parameters to control these strategies

        Args:
            raw_data (pd.DataFrame): The input DataFrame to be processed

        Returns:
            pd.DataFrame: The processed DataFrame (unchanged in this implementation)
        """

        if self.has_column_mappings:
            logger.info(
                "Converting data using FixedCombinationTransformer... Finished (No action)."
            )
            return raw_data

        logger.info("Converting data using FixedCombinationTransformer... ")

        # iterate through all simplified fixed combination relationships
        for base_col, related_cols in self.simplified_fixed_combinations.items():
            if base_col not in raw_data.columns:
                continue

            # get the unique values of the base column
            base_values = raw_data[base_col].unique()

            # process each related column
            for related_col in related_cols:
                if related_col not in raw_data.columns:
                    continue

                # creating a value mapping dictionary
                value_mapping = {}
                for base_val in base_values:
                    # create a value mapping dictionary
                    related_vals = raw_data[raw_data[base_col] == base_val][related_col].unique()
                    if len(related_vals) == 1:
                        value_mapping[base_val] = related_vals[0]

                # Save when the mapping dictionary is non-empty and the reference column is not entirely NaN.
                if value_mapping and not any(pd.isna(v) for v in value_mapping.values()):
                    self.column_mappings[(base_col, related_col)] = value_mapping
                    logger.debug(f"Saved mapping relationship between {base_col} and {related_col}")

        logger.info("Converting data using FixedCombinationTransformer... Finished.")

        self.has_column_mappings = True

        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the conversion process applied by the FixedCombinationTransformer.

        This method takes the processed DataFrame and uses the saved column mappings
        to restore the original values based on the relationships defined during the
        conversion process. If a base value does not have a corresponding related value,
        a random base value is selected to ensure that the DataFrame remains consistent.

        Args:
            processed_data (pd.DataFrame): The input DataFrame containing the processed data.

        Returns:
            pd.DataFrame: The DataFrame with original values restored based on the defined mappings.
        """

        result_df = processed_data.copy()

        logger.info("Reverse converting data using FixedCombinationTransformer...")

        # iterate through all column mappings
        for (base_col, related_col), mapping in self.column_mappings.items():
            if base_col not in result_df.columns or related_col not in result_df.columns:
                continue

            # define a function to replace base_col and related_col
            def replace_row(row):
                base_val = row[base_col]
                if base_val in mapping:
                    new_related_val = mapping[base_val]
                    return pd.Series({base_col: base_val, related_col: new_related_val})
                else:
                    # randomly select a base_val and get the corresponding related_val
                    new_base_val = random.choice(list(mapping.keys()))
                    new_related_val = mapping[new_base_val]
                    return pd.Series({base_col: new_base_val, related_col: new_related_val})

            # apply the `replace` function, and update the DataFrame
            replaced = result_df.apply(replace_row, axis=1)
            result_df[base_col] = replaced[base_col]
            result_df[related_col] = replaced[related_col]

        logger.info("Reverse converting data using FixedCombinationTransformer... Finished.")
        return result_df


@hookimpl
def register(manager):
    manager.register("FixedCombinationTransformer", FixedCombinationTransformer)
