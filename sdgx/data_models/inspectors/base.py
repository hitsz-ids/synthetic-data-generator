from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from sdgx.data_models.metadata import Metadata

from sdgx.data_models.relationship import Relationship


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


class RelationshipInspector(Inspector):
    """
    Empty RelationshipInspector for inheritence

    Subclass should implement `_build_relationship` and `fit`
    """

    def _build_relationship(self) -> list[Relationship]:
        return []

    def fit(
        self,
        raw_data: pd.DataFrame,
        name: str | None = None,
        metadata: "Metadata" | None = None,
        *args,
        **kwargs,
    ):
        pass

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""
        return {"relationships": self._build_relationship()}
