from typing import Any, Dict, List

from pydantic import BaseModel

from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship
from sdgx.exceptions import MultiTableCombinerError
from sdgx.utils import logger


class MultiTableCombiner(BaseModel):
    """MultiTableCombiner: combine different tables using relationship

    Args:
        metadata_dict (Dict[str, Any]):

        relationships (List[Any]):
    """

    metadata_version: str = "1.0"

    metadata_dict: Dict[str, Any] = {}
    relationships: List[Relationship] = []

    def check(self):
        """Do necessary checks:

        - Whether number of tables corresponds to relationships.
        - Whether table names corresponds to the relationship between tables;
        """

        # count check
        relationship_cnt = len(self.relationships)
        metadata_cnt = len(self.metadata_dict.keys())
        if metadata_cnt != relationship_cnt + 1:
            raise MultiTableCombinerError("Number of tables should corresponds to relationships.")

        table_names = set(self.metadata_dict.keys())
        relationship_parents = set(r.parent_table for r in self.relationships)
        relationship_children = set(r.child_table for r in self.relationships)

        # each relationship's table must have metadata
        if not table_names.issuperset(relationship_parents):
            raise MultiTableCombinerError(
                f"Relationships' parent table {relationship_parents - table_names} is missing."
            )
        if not table_names.issuperset(relationship_children):
            raise MultiTableCombinerError(
                f"Relationships' child table {relationship_children - table_names} is missing."
            )

        # each table in metadata must in a relationship
        if not (relationship_parents + relationship_children).issuperset(table_names):
            raise MultiTableCombinerError(
                f"Table {table_names - (relationship_parents+relationship_children)} is missing in relationships."
            )

        logger.info("MultiTableCombiner check finished.")
