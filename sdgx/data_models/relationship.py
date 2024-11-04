from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from typing import Any, Iterable, List, Union

from pydantic import BaseModel

from sdgx.exceptions import RelationshipInitError

KeyTuple = namedtuple("KeyTuple", ["parent", "child"])


class Relationship(BaseModel):
    """Relationship between tables

    For parent table, we don't need define primary key here.
    The primary key is pre-defined in parent table's metadata.

    Child table's foreign key should be defined here.
    """

    version: str = "1.0"

    # table names
    parent_table: str

    child_table: str

    foreign_keys: List[KeyTuple]
    """
    foreign keys.

    If key is a tuple, the first element is parent column name and the second element is child column name
    """

    @classmethod
    def build(
        cls,
        parent_table: str,
        child_table: str,
        foreign_keys: Iterable[str | tuple[str, str] | KeyTuple],
        parent_metadata: Metadata | None = None,
        child_metadata: Metadata | None = None,
    ) -> "Relationship":
        """
        Build relationship from parent table, child table and foreign keys

        Args:
            parent_table (str): parent table
            parent_metadata : metadata of parent table
            child_table (str): child table
            child_metadata : metadata of child table
            foreign_keys (Iterable[str | tuple[str, str]]): foreign keys. If key is a tuple, the first element is parent column name and the second element is child column name
        """

        if not parent_table:
            raise RelationshipInitError("parent table cannot be empty")
        if not child_table:
            raise RelationshipInitError("child table cannot be empty")

        foreign_keys = [
            KeyTuple(key, key) if isinstance(key, str) else KeyTuple(*key) for key in foreign_keys
        ]

        if not foreign_keys:
            raise RelationshipInitError("foreign keys cannot be empty")
        if parent_table == child_table:
            raise RelationshipInitError("child table and parent table cannot be the same")
        if parent_metadata and child_metadata:
            for key in foreign_keys:
                if type(parent_metadata) is not dict:
                    if key[0] not in parent_metadata.id_columns:
                        raise RelationshipInitError("type of foreign key in parent table is not id")
                    if key[1] not in child_metadata.id_columns:
                        raise RelationshipInitError("type of foreign key in child table is not id")
                else:  # if load from json file, Metadata is a dict
                    if key[0] not in parent_metadata["id_columns"]:
                        raise RelationshipInitError("type of foreign key in parent table is not id")
                    if key[1] not in child_metadata["id_columns"]:
                        raise RelationshipInitError("type of foreign key in child table is not id")
        return cls(
            parent_table=parent_table,
            child_table=child_table,
            foreign_keys=foreign_keys,
        )

    def _dump_json(self):
        return self.model_dump_json()

    def save(self, path: str | Path):
        """
        Save relationship to json file.
        """

        with path.open("w") as f:
            f.write(self._dump_json())

    @classmethod
    def load(cls, path: str | Path) -> "Relationship":
        """
        Load relationship from json file.
        """

        path = Path(path).expanduser().resolve()
        fields = json.load(path.open("r"))
        # print(fields)
        version = fields.pop("version", None)
        if version:
            cls.upgrade(version, fields)

        return Relationship.build(**fields)

    @classmethod
    def upgrade(cls, old_version: str, fields: dict[str, Any]) -> None:
        pass
