from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any

import pandas as pd

from sdgx.data_models.inspectors.base import RelationshipInspector
from sdgx.data_models.inspectors.extension import hookimpl
from sdgx.data_models.relationship import Relationship

if TYPE_CHECKING:
    from sdgx.data_models.metadata import Metadata


class SubsetRelationshipInspector(RelationshipInspector):
    """
    Inspecting relationships by comparing two columns is subset or not. So it needs to inspect all data for prev
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maybe_related_columns: dict[str, dict[str, pd.Series]] = {}

    def _is_related(self, p: pd.Series, c: pd.Series) -> bool:
        """
        If child is subset of parent, assume related
        """

        return c.isin(p).all()

    def _build_relationship(self) -> list[Relationship]:
        r = []
        for parent, p_m_related in self.maybe_related_columns.items():
            for child, c_m_related in self.maybe_related_columns.items():
                if parent == child:
                    continue
                related_pairs = []
                for p_col, p_df in p_m_related.items():
                    for c_col, c_df in c_m_related.items():
                        if self._is_related(p_df, c_df):
                            related_pairs.append((p_col, c_col) if p_col != c_col else p_col)
                if related_pairs:
                    r.append(Relationship.build(parent, child, related_pairs))
        return r

    def fit(
        self,
        raw_data: pd.DataFrame,
        name: str | None = None,
        metadata: "Metadata" | None = None,
        *args,
        **kwargs,
    ):
        columns = set(n for n in chain(metadata.id_columns, metadata.primary_keys))
        for c in columns:
            cur_map = self.maybe_related_columns.setdefault(name, dict())
            cur_map[c] = pd.concat(
                (cur_map.get(c, pd.Series()), raw_data[[c]].squeeze()),
                ignore_index=True,
            )

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""
        return {"relationships": self._build_relationship()}


@hookimpl
def register(manager):
    manager.register("SubsetRelationshipInspector", SubsetRelationshipInspector)
