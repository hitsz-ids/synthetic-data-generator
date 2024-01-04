from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List, Set, Tuple, Union

from pydantic import BaseModel

from sdgx.exceptions import RelationshipInitError


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

    foreign_keys: List[Union[str, Tuple[str, str]]]
    """
    foreign keys.

    If key is a tuple, the first element is parent column name and the second element is child column name
    """

    @classmethod
    def build(
        cls,
        parent_table: str,
        child_table: str,
        foreign_keys: Iterable[str | tuple[str, str]],
    ) -> "Relationship":
        """
        Build relationship from parent table, child table and foreign keys

        Args:
            parent_table (str): parent table
            child_table (str): child table
            foreign_keys (Iterable[str | tuple[str, str]]): foreign keys. If key is a tuple, the first element is parent column name and the second element is child column name
        """

        if not parent_table:
            raise RelationshipInitError("parent table cannot be empty")
        if not child_table:
            raise RelationshipInitError("child table cannot be empty")

        foreign_keys = list(foreign_keys)
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
        version = fields.pop("version", None)
        if version:
            cls.upgrade(version, fields)

        return Relationship.build(**fields)

    @classmethod
    def upgrade(cls, old_version: str, fields: dict[str, Any]) -> None:
        pass
