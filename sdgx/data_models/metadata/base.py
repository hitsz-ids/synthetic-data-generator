from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.exceptions import MetadataInitError
from sdgx.utils import logger

# TODO: Design metadata for relationships...
# class DType(Enum):
#     datetime = "datetime"
#     timestamp = "timestamp"
#     numeric = "numeric"
#     category = "category"


# class Relationship:
#     pass


class Metadata(BaseModel):
    discrete_columns: List[str] = []
    _extend: Dict[str, Any] = {}

    def get(self, key: str, default=None) -> Any:
        return getattr(self, key, getattr(self._extend, key, default))

    def set(self, key: str, value: Any):
        if key == "_extend":
            raise MetadataInitError("Cannot set _extend directly")

        if key in self.model_fields:
            setattr(self, key, value)
        else:
            self._extend[key] = value

    def update(self, attributes: dict[str, Any]):
        for k, v in attributes.items():
            self.set(k, v)

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
        logger.info("Inspecting metadata...")
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
            metadata.update(inspector.inspect())

        return metadata

    def save(self, path: str | Path):
        with path.open("w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def load(cls, path: str | Path) -> "Metadata":
        path = Path(path).expanduser().resolve()
        attributes = json.load(path.open("r"))
        return Metadata().update(attributes)
