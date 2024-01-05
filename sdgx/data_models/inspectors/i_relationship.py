from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from sdgx.data_models.inspectors.base import RelationshipInspector
from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.relationship import Relationship

if TYPE_CHECKING:
    from sdgx.data_models.metadata import Metadata


class DefaultRelationshipInspector(RelationshipInspector):
    """
    Inspecting relationships by Column similarity
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


@hookimpl
def register(manager):
    manager.register("DefaultRelationshipInspector", DefaultRelationshipInspector)
