from typing import List
from pydantic import BaseModel

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

    # foreign keys
    child_table_foreign_key: str = "foreign key undefined"
    
    # for composite keys 
    composite_foreign_key:bool = False
    child_table_composite_foreign_key: List[str] = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.parent_table == self.child_table:
            raise ValueError("child table and parent table cannot be the same")
