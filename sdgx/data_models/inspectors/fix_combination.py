# FixCombinationInspector
from __future__ import annotations

from typing import Any
import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl

class FixCombinationInspector(Inspector):
    """
    FixCombinationInspector is designed to identify columns in a DataFrame that have fixed relationships based on covariance.

    Attributes:
        fixed_combinations (dict[str, set[str]]): A dictionary mapping column names to sets of column names that have fixed relationships with them.
        _inspect_level (int): The inspection level for this inspector, set to 70.
    """

    fixed_combinations: dict[str, set[str]] = {}
    """
    A dictionary mapping column names to sets of column names that have fixed relationships with them.
    """

    _inspect_level = 70
    """
    The inspection level for this inspector, set to 70. This attribute indicates the priority or depth of inspection that this inspector performs relative to other inspectors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """
        Fit the inspector to the raw data.

        This method calculates the covariance matrix of the DataFrame, ignoring NaN values, and identifies columns that have fixed relationships.

        Args:
            raw_data (pd.DataFrame): The raw data to be inspected.

        Returns:
            None
        """
        # Calculate the covariance matrix, ignoring NaN values
        covariance_matrix = raw_data.dropna().cov()

        self.fixed_combinations = {}
        for column in covariance_matrix.columns:
            # Identify columns with high covariance (fixed relationships)
            related_columns = set(covariance_matrix.index[covariance_matrix[column].abs() > 0.9])
            related_columns.discard(column)  # Remove self-reference
            if related_columns:
                self.fixed_combinations[column] = related_columns

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"fixed_combinations": self.fixed_combinations}


@hookimpl
def register(manager):
    manager.register("FixCombinationInspector", FixCombinationInspector)