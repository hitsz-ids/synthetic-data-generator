from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.relationship import Relationship


class RelationshipInspector(Inspector):
    def _build_relationship(self) -> list[Relationship]:
        return []

    def fit(self, raw_data: pd.DataFrame, name: str, *args, **kwargs):
        pass

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""
        return {"relationships": self._build_relationship()}


@hookimpl
def register(manager):
    manager.register("DefaultRelationshipInspector", RelationshipInspector)
