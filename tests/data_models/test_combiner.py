from __future__ import annotations

import pandas as pd
import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.combiner import MetadataCombiner
from sdgx.data_models.inspectors.base import RelationshipInspector
from sdgx.data_models.relationship import Relationship


class MockInspector(RelationshipInspector):
    def __init__(self, dummy_data: list[Relationship], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tables = set()
        self.dummy_data = dummy_data

    def _build_relationship(self) -> list[Relationship]:
        return self.dummy_data


def test_from_dataloader(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pairs = demo_relational_table_path
    dl_a = DataLoader(CsvConnector(path=table_a_path))
    dl_b = DataLoader(CsvConnector(path=table_b_path))
    relationship = Relationship.build(
        parent_table=dl_a.identity,
        child_table=dl_b.identity,
        foreign_keys=pairs,
    )

    combiner = MetadataCombiner.from_dataloader(
        dataloaders=[dl_a, dl_b],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=MockInspector,
        relationships_inspector_kwargs=dict(dummy_data=[relationship]),
    )

    assert dl_a.identity in combiner.named_metadata
    assert dl_b.identity in combiner.named_metadata
    assert combiner.relationships == [relationship]

    save_dir = tmp_path / "unittest-combinner"
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner


def test_from_dataframe(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pair = demo_relational_table_path
    relationship = Relationship.build(
        parent_table="table_a",
        child_table="table_b",
        foreign_keys=pair,
    )

    tb_a = pd.read_csv(table_a_path)
    tb_b = pd.read_csv(table_b_path)

    combiner = MetadataCombiner.from_dataframe(
        dataframes=[tb_a, tb_b],
        names=["table_a", "table_b"],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=MockInspector,
        relationships_inspector_kwargs=dict(dummy_data=[relationship]),
    )

    assert "table_a" in combiner.named_metadata
    assert "table_b" in combiner.named_metadata
    assert combiner.relationships == [relationship]

    save_dir = tmp_path / "unittest-combinner"
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner


def test_custom_build_from_dataloaders(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pairs = demo_relational_table_path
    dl_a = DataLoader(CsvConnector(path=table_a_path))
    dl_b = DataLoader(CsvConnector(path=table_b_path))
    relationship = Relationship.build(
        parent_table=dl_a.identity,
        child_table=dl_b.identity,
        foreign_keys=pairs,
    )
    combiner = MetadataCombiner.from_dataloader(
        dataloaders=[dl_a, dl_b],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=MockInspector,
        relationships_inspector_kwargs=dict(
            dummy_data=Relationship.build(
                parent_table="balaP",
                child_table="balaC",
                foreign_keys=["balabala"],
            )
        ),
        relationships=[relationship],
    )
    assert dl_a.identity in combiner.named_metadata
    assert dl_b.identity in combiner.named_metadata
    assert combiner.relationships == [relationship]

    save_dir = tmp_path / "unittest-combinner"
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner


def test_custom_build_from_dataframe(demo_relational_table_path, tmp_path):
    table_a_path, table_b_path, pair = demo_relational_table_path
    relationship = Relationship.build(
        parent_table="table_a",
        child_table="table_b",
        foreign_keys=pair,
    )
    tb_a = pd.read_csv(table_a_path)
    tb_b = pd.read_csv(table_b_path)

    combiner = MetadataCombiner.from_dataframe(
        dataframes=[tb_a, tb_b],
        names=["table_a", "table_b"],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=MockInspector,
        relationships_inspector_kwargs=dict(
            dummy_data=Relationship.build(
                parent_table="balaP",
                child_table="balaC",
                foreign_keys=["balabala"],
            )
        ),
        relationships=[relationship],
    )
    assert "table_a" in combiner.named_metadata
    assert "table_b" in combiner.named_metadata
    assert combiner.relationships == [relationship]

    save_dir = tmp_path / "unittest-combinner"
    combiner.save(save_dir)
    assert save_dir.exists()
    loaded_combiner = MetadataCombiner.load(save_dir)
    assert combiner == loaded_combiner


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
