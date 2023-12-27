from typing import List

from pydantic import BaseModel

from sdgx.exceptions import RelationshipError


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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.parent_table == self.child_table:
            raise RelationshipError("child table and parent table cannot be the same")
