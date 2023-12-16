from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.inspect_meta import InspectMeta
from sdgx.data_models.inspectors.manager import InspectorManager


class DType(Enum):
    datetime = "datetime"
    timestamp = "timestamp"
    numeric = "numeric"
    category = "category"


class Relationship:
    pass


class Metadata(BaseModel):
    # fields: List[str]

    def update(self, inspect_meta: InspectMeta):
        return self

    @classmethod
    def from_dataloader(
        cls,
        dataloader: DataLoader,
        max_chunk: int = 10,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
    ) -> "Metadata":
        inspectors = InspectorManager().init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for i, chunk in enumerate(dataloader.iter()):
            for inspector in inspectors:
                inspector.fit(chunk)
            if all(i.ready for i in inspectors) or i > max_chunk:
                break

        metadata = Metadata()
        for inspector in inspectors:
            metadata.update(inspector.inspect())

        return metadata

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
    ) -> "Metadata":
        inspectors = InspectorManager().init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for inspector in inspectors:
            inspector.fit(df)

        metadata = Metadata()
        for inspector in inspectors:
            if inspector.ready:
                metadata.update(inspector.inspect())

        return metadata
