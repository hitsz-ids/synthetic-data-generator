from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship
from sdgx.exceptions import MultiTableCombinerError
from sdgx.utils import logger


class MetadataCombiner(BaseModel):
    """
    Combine different tables with relationship, used for describing the relationship between tables.

    Args:
        named_metadata (Dict[str, Any]): Name of the table: Metadata

        relationships (List[Any])
    """

    metadata_version: str = "1.0"

    named_metadata: Dict[str, Metadata] = {}

    relationships: List[Relationship] = []

    def check(self):
        """Do necessary checks:

        - Whether number of tables corresponds to relationships.
        - Whether table names corresponds to the relationship between tables;
        """
        for m in self.named_metadata.values():
            m.check()

        relationship_cnt = len(self.relationships)
        metadata_cnt = len(self.named_metadata.keys())
        if metadata_cnt != relationship_cnt + 1:
            raise MultiTableCombinerError("Number of tables should corresponds to relationships.")

        table_names = set(self.named_metadata.keys())
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

    @classmethod
    def from_dataloader(
        cls,
        dataloaders: list[DataLoader],
        max_chunk: int = 10,
        metadata_from_dataloader_kwargs: None | dict = None,
        relationshipe_inspector: None | str | type[Inspector] = "DefaultRelationshipInspector",
        relationships_inspector_kwargs: None | dict = None,
        relationships: None | list[Relationship] = None,
    ):
        if not isinstance(dataloaders, list):
            dataloaders = [dataloaders]

        metadata_from_dataloader_kwargs = metadata_from_dataloader_kwargs or {}
        named_metadata = {
            d.identity: Metadata.from_dataloader(d, **metadata_from_dataloader_kwargs)
            for d in dataloaders
        }

        if relationships is None and relationshipe_inspector is not None:
            if relationships_inspector_kwargs is None:
                relationships_inspector_kwargs = {}

            inspector = InspectorManager().init(
                relationshipe_inspector, **relationships_inspector_kwargs
            )
            for d in dataloaders:
                for i, chunk in enumerate(d.iter()):
                    inspector.fit(chunk)
                    if inspector.ready or i > max_chunk:
                        break
            relationships = inspector.inspect()["relationships"]

        return cls(named_metadata=named_metadata, relationships=relationships)

    @classmethod
    def from_dataframe(
        cls,
        dataframes: list[pd.DataFrame],
    ) -> "MetadataCombiner":
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        ...

    def _dump_json(self):
        return self.model_dump_json()

    def save(
        self,
        save_dir: str | Path,
        metadata_subdir: str = "metadata",
        relationship_subdir: str = "relationship",
    ):
        save_dir = Path(save_dir).expanduser().resolve()
        for name, metadata in self.named_metadata.items():
            metadata.save(save_dir / metadata_subdir / f"{name}.json")

        for relationship in self.relationships:
            relationship.save(
                save_dir
                / relationship_subdir
                / f"{relationship.parent_table}_{relationship.child_table}.json"
            )

    @classmethod
    def load(
        cls,
        save_dir: str | Path,
        metadata_subdir: str = "metadata",
        relationship_subdir: str = "relationship",
    ) -> "MetadataCombiner":
        save_dir = Path(save_dir).expanduser().resolve()

        named_metadata = {p.stem: Metadata.load(p) for p in (save_dir / metadata_subdir).glob("*")}

        relationships = [Relationship.load(p) for p in (save_dir / relationship_subdir).glob("*")]

        return cls(named_metadata=named_metadata, relationships=relationships)
