from __future__ import annotations

from typing import Any

import pandas as pd


class Inspector:
    """
    Base Inspector class

    Inspector is used to inspect data and generate metadata automatically.

    Parameters:
        ready (bool): Ready to inspect, maybe all fields are fitted, or indicate if there is more data, inspector will be more precise.
    """

    def __init__(self, *args, **kwargs):
        self.ready: bool = False

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """Fit the inspector.

        Args:
            raw_data (pd.DataFrame): Raw data
        """
        return

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""
