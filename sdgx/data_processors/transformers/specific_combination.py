from __future__ import annotations

from typing import Dict, List, Set

import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger

"""
Make sure that user's specific combination are correct in sample data.
"""


class SpecificCombinationTransformer(Transformer):
    """
    Define a list where each element is a set containing string type column names
    """

    column_groups: List[Set[str]]

    """
    Define a dictionary variable `mappings` where the keys are frozensets and the values are pandas DataFrame objects
    """
    mappings: Dict[frozenset, pd.DataFrame]

    def __init__(self):
        self.column_groups: List[Set[str]] = []
        self.mappings: Dict[frozenset, pd.DataFrame] = {}

    def fit(self, metadata: Metadata | None = None, tabular_data: DataLoader | pd.DataFrame = None):
        """
        Study the combination relationships and value mapping of columns.
        """

        # TODO-2024/11/14 - Defined how metadata passing the specific combination
        df = tabular_data
        column_lists = metadata.get("specific_combinations")
        if column_lists is None or len(column_lists) == 0:
            logger.warning("No specific combination information found in metadata.")
            return

        self.column_groups = [set(cols) for cols in column_lists]

        # Create a mapping relationship for each group of columns
        for group in self.column_groups:
            group_df = df[list(group)].drop_duplicates()
            self.mappings[frozenset(group)] = group_df

        self.fitted = True
        logger.info("SpecificCombinationTransformer Fitted.")

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("SpecificCombinationTransformer convert doing nothing...")
        return super().convert(raw_data)

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure that the data conforms to the learned mapping relationships
        """
        result_df = processed_data.copy()

        for group in self.column_groups:
            group_mapping = self.mappings[frozenset(group)]
            group_cols = list(group)

            # Check and correct each row
            for idx in range(len(result_df)):
                row_values = result_df.loc[idx, group_cols].to_dict()

                # Find the best matching row in the mapping table
                best_match = None
                max_matches = -1

                for _, mapping_row in group_mapping.iterrows():
                    matches = sum(row_values[col] == mapping_row[col] for col in group_cols)
                    if matches > max_matches:
                        max_matches = matches
                        best_match = mapping_row

                # Update the current row using the best match
                if best_match is not None:
                    result_df.loc[idx, group_cols] = best_match

        return result_df


@hookimpl
def register(manager):
    manager.register("SpecificCombinationTransformer", SpecificCombinationTransformer)
