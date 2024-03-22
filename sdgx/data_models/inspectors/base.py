from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from sdgx.data_models.metadata import Metadata

from sdgx.data_models.relationship import Relationship
from sdgx.exceptions import InspectorInitError


class Inspector:
    """
    Base Inspector class

    Inspector is used to inspect data and generate metadata automatically.

    Parameters:
        ready (bool): Ready to inspect, maybe all fields are fitted, or indicate if there is more data, inspector will be more precise.
    """

    pii = False
    """
    PII refers if a column contains private or sensitive information.
    """

    _inspect_level: int = 10
    """
    Private variable used to store property inspect_level's value.
    """

    ready: bool = False
    """
    Indicates whether the inspector has completed its inference.

    When completed, ready == True.
    """

    @property
    def inspect_level(self):
        """
        Inspected level is a concept newly introduced in version 0.1.6. Since a single column in the table may be marked by different inspectors at the same time (for example: the email column may be recognized as email, but it may also be recognized as the id column, and it may also be recognized by different inspectors at the same time identified as a discrete column, which will cause confusion in subsequent processing), the inspect_leve is used when determining the specific type of a column.

        We will preset different inspector levels for different inspectors, usually more specific inspectors will get higher levels, and general inspectors (like discrete) will have inspect_level.

        The value of the variable inspect_level is limited to 1-100. In baseclass and bool, discrete and numeric types, the inspect_level is set to 10. For datetime and id types, the inspect_level is set to 20.

        Current inspect_level value will make it easier for developers to insert a custom inspector from the middle.
        """
        return self._inspect_level

    @inspect_level.setter
    def inspect_level(self, value: int):
        if value > 0 and value <= 100:
            self._inspect_level = value
        else:
            raise InspectorInitError("The inspect_level should be set in [1, 100].")

    def __init__(self, inspect_level=None, *args, **kwargs):
        self.ready: bool = False
        # add inspect_level check
        if inspect_level:
            self.inspect_level = inspect_level

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
