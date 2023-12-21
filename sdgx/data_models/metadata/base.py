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


class Metadata(BaseModel):
    """Metadata

    This metadata is mainly used to describe the data types of all columns in a single data table.

    For each column, there should be an instance of the Data Type object.
    """

    primary_key: str
    column_list: list[str]

    discrete_columns: List[str] = []
    _extend: Dict[str, Any] = {}

    composite_primary_key: bool = False
    primary_key_list: list = []

    metadata_version: str = "1.0"

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
        primary_key: str = None,
        include_inspectors: list[str] | None = None,
        exclude_inspectors: list[str] | None = None,
        inspector_init_kwargs: dict[str, Any] | None = None,
    ) -> "Metadata":
        """Initialize a metadata from DataLoader and Inspectors

        Args:
            dataloader(DataLoader): the input DataLoader.

            max_chunk(int): max chunk count.

            primary_key(list(str) | str): the primary key of this table, use the first column in table by default.

            include_inspectors(list[str]): data type inspectors that should included in this metadata (table).

            exclude_inspectors(list[str]): data type inspectors that should NOT included in this metadata (table).

            inspector_init_kwargs(dict): inspector args.
        """
        logger.info("Inspecting metadata...")
        inspectors = InspectorManager().init_inspcetors(
            include_inspectors, exclude_inspectors, **(inspector_init_kwargs or {})
        )
        for i, chunk in enumerate(dataloader.iter()):
            for inspector in inspectors:
                inspector.fit(chunk)
            if all(i.ready for i in inspectors) or i > max_chunk:
                break
        # If primary_key is not specified, it defaults to the first column of the table.
        # Primary keys can be updated after instantiation.
        if primary_key is None:
            primary_key = dataloader.columns()[0]

        metadata = Metadata(primary_key=primary_key, column_list=dataloader.columns())
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

        metadata = Metadata(primary_key=df.columns[0], column_list=list(df.columns))
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
