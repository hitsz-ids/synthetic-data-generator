from __future__ import annotations

import json
from pathlib import Path
from typing import List

from pydantic import BaseModel

from sdgx.exceptions import RelationshipInitError


class Relationship(BaseModel):
    """Relationship between tables

    For parent table, we don't need define primary key here.
    The primary key is pre-defined in parent table's metadata.

    Child table's foreign key should be defined here.
    """

    metadata_version: str = "1.0"

    # table names
    parent_table: str
    child_table: str

    foreign_keys: List[str] = []

    @classmethod
    def build(cls, parent_table: str, child_table: str, foreign_keys: List[str]) -> "Relationship":
        if not parent_table:
            raise RelationshipInitError("parent table cannot be empty")
        if not child_table:
            raise RelationshipInitError("child table cannot be empty")
        if not foreign_keys:
            raise RelationshipInitError("foreign keys cannot be empty")
        if parent_table == child_table:
            raise RelationshipInitError("child table and parent table cannot be the same")

        return cls(
            parent_table=parent_table,
            child_table=child_table,
            foreign_keys=foreign_keys,
        )

    def _dump_json(self):
        return self.model_dump_json()

    def save(self, path: str | Path):
        with path.open("w") as f:
            f.write(self._dump_json())

    @classmethod
    def load(cls, path: str | Path) -> "Relationship":
        path = Path(path).expanduser().resolve()
        fields = json.load(path.open("r"))
        return Relationship.build(**fields)
