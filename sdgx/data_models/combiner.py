from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship
from sdgx.exceptions import MetadataCombinerInitError, MetadataCombinerInvalidError
from sdgx.utils import logger


class MetadataCombiner(BaseModel):
    """
    Combine different tables with relationship, used for describing the relationship between tables.

    Args:
        version (str): version
        named_metadata (Dict[str, Any]): pairs of table name and metadata
        relationships (List[Any]): list of relationships
    """

    version: str = "1.0"

    named_metadata: Dict[str, Metadata] = {}

    relationships: List[Relationship] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def check(self):
        """Do necessary checks:

        - Whether number of tables corresponds to relationships.
        - Whether table names corresponds to the relationship between tables;
        """
        for m in self.named_metadata.values():
            m.check()

        table_names = set(self.named_metadata.keys())
        relationship_parents = set(r.parent_table for r in self.relationships)
        relationship_children = set(r.child_table for r in self.relationships)

        # each relationship's table must have metadata
        if not table_names.issuperset(relationship_parents):
            raise MetadataCombinerInvalidError(
                f"Relationships' parent table {relationship_parents - table_names} is missing."
            )
        if not table_names.issuperset(relationship_children):
            raise MetadataCombinerInvalidError(
                f"Relationships' child table {relationship_children - table_names} is missing."
            )

        # each table in metadata must in a relationship
        if not (relationship_parents | relationship_children).issuperset(table_names):
            raise MetadataCombinerInvalidError(
                f"Table {table_names - (relationship_parents+relationship_children)} is missing in relationships."
            )

        logger.info("MultiTableCombiner check finished.")

    @classmethod
    def from_dataloader(
        cls,
        dataloaders: list[DataLoader],
        metadata_from_dataloader_kwargs: None | dict = None,
        relationshipe_inspector: None | str | type[Inspector] = "SubsetRelationshipInspector",
        relationships_inspector_kwargs: None | dict = None,
        relationships: None | list[Relationship] = None,
    ):
        """
        Combine multiple dataloaders with relationship.

        Args:
            dataloaders (list[DataLoader]): list of dataloaders
            max_chunk (int): max chunk count for relationship inspector.
            metadata_from_dataloader_kwargs (dict): kwargs for :func:`Metadata.from_dataloader`
            relationshipe_inspector (str | type[Inspector]): relationship inspector
            relationships_inspector_kwargs (dict): kwargs for :func:`InspectorManager.init`
            relationships (list[Relationship]): list of relationships
        """
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
                for chunk in d.iter():
                    inspector.fit(
                        chunk,
                        name=d.identity,
                        metadata=named_metadata[d.identity],
                    )
            relationships = inspector.inspect()["relationships"]

        return cls(named_metadata=named_metadata, relationships=relationships)

    @classmethod
    def from_dataframe(
        cls,
        dataframes: list[pd.DataFrame],
        names: list[str],
        metadata_from_dataloader_kwargs: None | dict = None,
        relationshipe_inspector: None | str | type[Inspector] = "SubsetRelationshipInspector",
        relationships_inspector_kwargs: None | dict = None,
        relationships: None | list[Relationship] = None,
    ) -> "MetadataCombiner":
        """
        Combine multiple dataframes with relationship.

        Args:
            dataframes (list[pd.DataFrame]): list of dataframes
            names (list[str]): list of names
            metadata_from_dataloader_kwargs (dict): kwargs for :func:`Metadata.from_dataloader`
            relationshipe_inspector (str | type[Inspector]): relationship inspector
            relationships_inspector_kwargs (dict): kwargs for :func:`InspectorManager.init`
            relationships (list[Relationship]): list of relationships
        """
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        if not isinstance(names, list):
            names = [names]

        metadata_from_dataloader_kwargs = metadata_from_dataloader_kwargs or {}

        if len(dataframes) != len(names):
            raise MetadataCombinerInitError("dataframes and names should have same length.")

        named_metadata = {
            n: Metadata.from_dataframe(d, **metadata_from_dataloader_kwargs)
            for n, d in zip(names, dataframes)
        }

        if relationships is None and relationshipe_inspector is not None:
            if relationships_inspector_kwargs is None:
                relationships_inspector_kwargs = {}

            inspector = InspectorManager().init(
                relationshipe_inspector, **relationships_inspector_kwargs
            )
            for n, d in zip(names, dataframes):
                inspector.fit(
                    d,
                    name=n,
                    metadata=named_metadata[n],
                )
            relationships = inspector.inspect()["relationships"]

        return cls(named_metadata=named_metadata, relationships=relationships)

    def _dump_json(self):
        return self.model_dump_json()

    def save(
        self,
        save_dir: str | Path,
        metadata_subdir: str = "metadata",
        relationship_subdir: str = "relationship",
    ):
        """
        Save metadata to json file.

        This will create several subdirectories for metadata and relationship.

        Args:
            save_dir (str | Path): directory to save
            metadata_subdir (str): subdirectory for metadata, default is "metadata"
            relationship_subdir (str): subdirectory for relationship, default is "relationship"
        """
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        version_file = save_dir / "version"
        version_file.write_text(self.version)

        metadata_subdir = save_dir / metadata_subdir
        relationship_subdir = save_dir / relationship_subdir

        metadata_subdir.mkdir(parents=True, exist_ok=True)
        for name, metadata in self.named_metadata.items():
            metadata.save(metadata_subdir / f"{name}.json")

        relationship_subdir.mkdir(parents=True, exist_ok=True)
        for relationship in self.relationships:
            relationship.save(
                relationship_subdir / f"{relationship.parent_table}_{relationship.child_table}.json"
            )

    @classmethod
    def load(
        cls,
        save_dir: str | Path,
        metadata_subdir: str = "metadata",
        relationship_subdir: str = "relationship",
        version: None | str = None,
    ) -> "MetadataCombiner":
        """
        Load metadata from json file.

        Args:
            save_dir (str | Path): directory to save
            metadata_subdir (str): subdirectory for metadata, default is "metadata"
            relationship_subdir (str): subdirectory for relationship, default is "relationship"
            version (str): Manual version, if not specified, try to load from version file
        """

        save_dir = Path(save_dir).expanduser().resolve()
        if not version:
            logger.debug("No version specified, try to load from version file.")
            version_file = save_dir / "version"
            if version_file.exists():
                version = version_file.read_text().strip()
            else:
                logger.info("No version file found, assume version is 1.0")
                version = "1.0"

        named_metadata = {p.stem: Metadata.load(p) for p in (save_dir / metadata_subdir).glob("*")}

        relationships = [Relationship.load(p) for p in (save_dir / relationship_subdir).glob("*")]

        cls.upgrade(version, named_metadata, relationships)

        return cls(
            version=version,
            named_metadata=named_metadata,
            relationships=relationships,
        )

    @classmethod
    def upgrade(
        cls,
        old_version: str,
        named_metadata: dict[str, Metadata],
        relationships: list[Relationship],
    ) -> None:
        """
        Upgrade metadata from old version to new version

        :ref:`Metadata.upgrade` and :ref:`Relationship.upgrade` will try upgrade when loading.
        So here we just do Combiner's upgrade.
        """

        pass

    @property
    def fields(self) -> Iterable[str]:
        """
        Return all fields in MetadataCombiner.
        """

        return chain(
            (k for k in self.model_fields if k.endswith("_columns")),
        )

    def __eq__(self, other):
        if not isinstance(other, MetadataCombiner):
            return super().__eq__(other)

        # if self and other has the same
        return (
            self.version == other.version
            and all(
                self.get(key) == other.get(key) for key in set(chain(self.fields, other.fields))
            )
            and set(self.fields) == set(other.fields)
        )
