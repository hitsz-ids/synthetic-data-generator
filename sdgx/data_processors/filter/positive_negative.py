from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.filter.base import Filter
from sdgx.utils import logger


class PositiveNegativeFilter(Filter):
    """
    A data processor for filtering positive and negative values.

    This filter is used to ensure that values in specific columns remain positive or negative.
    During the reverse conversion process, rows that do not meet the expected positivity or
    negativity will be removed.

    Attributes:
        int_columns (set): A set of column names containing integer values.
        float_columns (set): A set of column names containing float values.
        positive_columns (set): A set of column names that should contain positive values.
        negative_columns (set): A set of column names that should contain negative values.
    """

    int_columns: set
    """
    A set of column names that contain integer values.
    """

    float_columns: set
    """
    A set of column names that contain float values.
    """

    positive_columns: set
    """
    A set of column names that are identified as containing positive numeric values.
    """

    negative_columns: set
    """
    A set of column names that are identified as containing negative numeric values.
    """

    def __init__(self):
        self.int_columns = set()
        self.float_columns = set()
        self.positive_columns = set()
        self.negative_columns = set()

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the data filter.
        """
        logger.info("PositiveNegativeFilter Fitted.")

        # record int and float data
        self.int_columns = metadata.int_columns
        self.float_columns = metadata.float_columns

        # record pos and neg
        self.positive_columns = set(metadata.numeric_format["positive"])
        self.negative_columns = set(metadata.numeric_format["negative"])

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method for data filter (No Action).
        """

        logger.info("Converting data using PositiveNegativeFilter... Finished (No Action)")

        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the pos_neg data filter.

        Iterate through each row of data, check if there are negative values in positive_columns,
        or positive values in negative_columns. If the conditions are not met, discard the row.
        """
        logger.info(
            f"Data reverse-converted by PositiveNegativeFilter Start with Shape: {processed_data.shape}."
        )

        # Create a boolean mask to mark the rows that need to be retained
        mask = pd.Series(True, index=processed_data.index)

        # Check positive_columns
        for col in self.positive_columns:
            if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col]):
                mask &= processed_data[col] >= 0

        # Check negative_columns
        for col in self.negative_columns:
            if col in processed_data.columns and pd.api.types.is_numeric_dtype(processed_data[col]):
                mask &= processed_data[col] <= 0

        # Apply the mask to filter the data
        filtered_data = processed_data[mask]

        logger.info(
            f"Data reverse-converted by PositiveNegativeFilter with Output Shape: {filtered_data.shape}."
        )

        return filtered_data


@hookimpl
def register(manager):
    manager.register("PositiveNegativeFilter", PositiveNegativeFilter)
