import pandas as pd

from sdgx.data_models.metadata import Metadata


class Inspector:
    """
    Base Inspector class

    Inspector is used to inspect data and generate metadata automatically.
    """

    ready: bool
    """Ready to inspect, maybe all fields are fitted."""

    def fit(self, raw_data: pd.DataFrame):
        """Fit the inspector.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        return

    def inspect(self) -> Metadata:
        """Inspect raw data and generate metadata."""
