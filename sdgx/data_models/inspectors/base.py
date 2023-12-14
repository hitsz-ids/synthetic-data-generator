import pandas as pd


class Inspector:
    """
    Base Inspector class

    Inspector is used to inspect data and generate metadata automatically.
    """

    def fit(self, raw_data: pd.DataFrame):
        """Fit the inspector.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        return

    def inspect(self):
        """Inspect raw data and generate metadata."""
