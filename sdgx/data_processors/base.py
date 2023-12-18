from __future__ import annotations

import pandas as pd

from sdgx.data_models.metadata import Metadata


class DataProcessor:
    """
    Base class for data processors.
    """

    def fit(self, metadata: Metadata | None = None):
        """Fit the data processor.

        Called before ``convert`` and ``reverse_convert``.

        Args:
            metadata (Metadata, optional): Metadata. Defaults to None.
        """
        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert raw data into processed data.

        Args:
            raw_data (pd.DataFrame): Raw data

        Returns:
            pd.DataFrame: Processed data
        """
        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Convert processed data into raw data.

        Args:
            processed_data (pd.DataFrame): Processed data

        Returns:
            pd.DataFrame: Raw data
        """
        return processed_data
