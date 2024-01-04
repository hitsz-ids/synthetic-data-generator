from __future__ import annotations

import pandas as pd
import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.combiner import MetadataCombiner
from sdgx.data_models.inspectors.relationship import RelationshipInspector
from sdgx.data_models.relationship import Relationship


class MockInspector(RelationshipInspector):
    def __init__(self, dummy_data: list[Relationship], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tables = set()
        self.dummy_data = dummy_data

    def _build_relationship(self) -> list[Relationship]:
        return self.dummy_data


def test_from_dataloader(demo_relational_table_path):
    table_a_path, table_b_path, pairs = demo_relational_table_path
    dl_a = DataLoader(CsvConnector(path=table_a_path))
    dl_b = DataLoader(CsvConnector(path=table_b_path))
    relationship = Relationship.build(
        parent_table=dl_a.identity,
        child_table=dl_b.identity,
        foreign_keys=pairs,
    )

    inspector = MockInspector(dummy_data=[relationship])
    combiner = MetadataCombiner.from_dataloader(
        dataloaders=[dl_a, dl_b],
        max_chunk=10,
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=inspector,
        relationships_inspector_kwargs={},
    )

    assert dl_a.identity in combiner.named_metadata
    assert dl_b.identity in combiner.named_metadata
    assert combiner.relationships == [relationship]


def test_from_dataframe(demo_relational_table_path):
    table_a_path, table_b_path, pair = demo_relational_table_path
    relationship = Relationship.build(
        parent_table="table_a",
        child_table="table_b",
        foreign_keys=pair,
    )
    inspector = MockInspector(dummy_data=pair)

    combiner = MetadataCombiner.from_dataframe(
        dataframes=[table_a_path, table_b_path],
        names=["table_a", "table_b"],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=inspector,
        relationships_inspector_kwargs={},
    )

    assert "table_a" in combiner.named_metadata
    assert "table_b" in combiner.named_metadata
    assert combiner.relationships == [relationship]


def test_custom_build_from_dataframe(demo_relational_table_path):
    table_a_path, table_b_path, pairs = demo_relational_table_path
    dl_a = DataLoader(CsvConnector(path=table_a_path))
    dl_b = DataLoader(CsvConnector(path=table_b_path))
    relationship = Relationship.build(
        parent_table=dl_a.identity,
        child_table=dl_b.identity,
        foreign_keys=pairs,
    )
    inspector = MockInspector(
        dummy_data=Relationship.build(
            parent_table="balaP", child_table="balaC", foreign_keys=["balabala"]
        )
    )
    combiner = MetadataCombiner.from_dataframe(
        dataloaders=[],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=inspector,
        relationships_inspector_kwargs={},
        relationships=relationship,
    )
    assert dl_a.identity in combiner.named_metadata
    assert dl_b.identity in combiner.named_metadata
    assert combiner.relationships == [relationship]


def test_custom_build_from_dataframe(demo_relational_table_path):
    table_a_path, table_b_path, pair = demo_relational_table_path
    relationship = Relationship.build(
        parent_table="table_a",
        child_table="table_b",
        foreign_keys=pair,
    )
    inspector = MockInspector(
        dummy_data=Relationship.build(
            parent_table="balaP", child_table="balaC", foreign_keys=["balabala"]
        )
    )
    combiner = MetadataCombiner.from_dataframe(
        dataframes=[table_a_path, table_b_path],
        names=["table_a", "table_b"],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=inspector,
        relationships_inspector_kwargs={},
        relationships=[relationship],
    )
    assert "table_a" in combiner.named_metadata
    assert "table_b" in combiner.named_metadata
    assert combiner.relationships == [relationship]


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
