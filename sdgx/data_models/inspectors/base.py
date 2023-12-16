import pandas as pd

from sdgx.data_models.inspectors.inspect_meta import InspectMeta


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

    def inspect(self) -> InspectMeta:
        """Inspect raw data and generate metadata."""
