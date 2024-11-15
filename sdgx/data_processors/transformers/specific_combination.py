from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class SpecificCombinationTransformer(Transformer):
    """
    A transformer used to handle specific combinations of columns in tabular data.

    The relationships between columns can be quite complex. Currently, we introduced `FixedCombinationTransformer`
    is not capable of comprehensive automatic detection. This transformer allows users to manually specify the
    mapping relationships between columns, specifically for multiple corresponding relationships. Users can define
    multiple groups, with each group supporting multiple columns. The transformer will record the combination values
    of each column, and in the `reverse_convert()`, it will restore any mismatched combinations from the recorded
    relationships.

    For example:

    | Category A | Category B | Category C | Category D | Category E |
    | :--------: | :--------: | :--------: | :--------: | :--------: |
    |     A1     |     B1     |     C1     |     D1     |     E1     |
    |     A1     |     B1     |     C2     |     D2     |     E2     |
    |     A2     |     B2     |     C1     |     D1     |     E3     |

    Here user can specific combination like (Category A, Category B), (Category C, Category D, Category E).

    For now, the `specific_combinations` passing by `Metadata`

    """

    column_groups: List[Set[str]]
    """
    Define a list where each element is a set containing string type column names
    """

    mappings: Dict[frozenset, pd.DataFrame]
    """
    Define a dictionary variable `mappings` where the keys are frozensets and the values are pandas DataFrame objects
    """

    specified: bool
    """
    Define a boolean that flag if user specified the combination, if true, that handle the `specific_combinations`
    """

    def __init__(self):
        self.column_groups: List[Set[str]] = []
        self.mappings: Dict[frozenset, pd.DataFrame] = {}
        self.specified = False

    def fit(self, metadata: Metadata | None = None, tabular_data: DataLoader | pd.DataFrame = None):
        """
        Study the combination relationships and value mapping of columns.

        Args:
            metadata: Metadata containing information about specific column combinations.
            tabular_data: The tabular data to be fitted, can be a DataLoader object or a pandas DataFrame.
        """
        specific_combinations = metadata.get("specific_combinations")
        if specific_combinations is None or len(specific_combinations) == 0:
            logger.info(
                "Fit data using SpecificCombinationTransformer(No specified)... Finished (No action)."
            )
            self.fitted = True
            return

        # Create a mapping relationship for each group of columns
        df = tabular_data
        self.column_groups = [set(cols) for cols in specific_combinations]
        for group in self.column_groups:
            group_df = df[list(group)].drop_duplicates()
            self.mappings[frozenset(group)] = group_df

        self.fitted = True
        self.specified = True
        logger.info("SpecificCombinationTransformer Fitted.")

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the raw data based on the learned mapping relationships.

        Args:
           raw_data: The raw data to be converted.

        Returns:
           The converted data.
        """
        if not self.specified:
            logger.info(
                "Converting data using SpecificCombinationTransformer(No specified)... Finished (No action)."
            )
            return super().convert(raw_data)

        logger.info("SpecificCombinationTransformer convert doing nothing...")
        return super().convert(raw_data)

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse convert the processed data to ensure it conforms to the original format.

        Args:
            processed_data: The processed data to be reverse converted.

        Returns:
            The reverse converted data.
        """
        if not self.specified:
            logger.info(
                "Reverse converting data using SpecificCombinationTransformer(No specified)... Finished (No action)."
            )
            return processed_data

        result_df = processed_data.copy()
        n_rows = len(result_df)

        # For each column-mapping groups, replace with random choice
        # Here we random_indices for len(processed_data) from column-mapping and replaced processed_data.
        for group in self.column_groups:
            group_mapping = self.mappings[frozenset(group)]
            group_cols = list(group)
            random_indices = np.random.choice(len(group_mapping), size=n_rows)
            random_mappings = group_mapping.iloc[random_indices]
            result_df[group_cols] = random_mappings[group_cols].values

        return result_df


@hookimpl
def register(manager):
    manager.register("SpecificCombinationTransformer", SpecificCombinationTransformer)
