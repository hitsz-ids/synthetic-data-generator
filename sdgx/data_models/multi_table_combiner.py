from typing import Any, Dict, List

from pydantic import BaseModel

from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship


class MultiTableCombiner(BaseModel):
    """MultiTableCombiner: combine different tables using relationship

    Args:
        metadata_dict (Dict[str, Any]):

        relationships (List[Any]):
    """

    metadata_version: str = "1.0"

    metadata_dict: Dict[str, Any] = {}
    relationships: List[Any] = []

    def check(self):
        """Do necessary checks:

        - Whether number of tables corresponds to relationships.
        - Whether table names corresponds to the relationship between tables;
        """

        # count check
        relationship_cnt = len(self.relationships)
        metadata_cnt = len(self.metadata_dict.keys())
        if metadata_cnt != relationship_cnt + 1:
            raise ValueError("Number of tables should corresponds to relationships.")

        # table name check
        table_names_from_relationships = set()

        # each relationship's table must have metadata
        table_names = list(self.metadata_dict.keys())
        for each_r in self.relationships:
            if each_r.parent_table not in table_names:
                raise ValueError(f"Metadata of parent table {each_r.parent_table} is missing.")
            if each_r.child_table not in table_names:
                raise ValueError(f"Metadata of child table {each_r.child_table} is missing.")
            table_names_from_relationships.add(each_r.parent_table)
            table_names_from_relationships.add(each_r.child_table)

        # each table in metadata must in a relationship
        for each_t in table_names:
            if each_t not in table_names_from_relationships:
                raise ValueError(f"Table {each_t} has not relationship.")

        return True
