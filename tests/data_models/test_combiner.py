from __future__ import annotations

import pytest

from sdgx.data_models.combiner import MetadataCombiner
from sdgx.data_models.inspectors.relationship import RelationshipInspector
from sdgx.data_models.relationship import Relationship


class MockInspector(RelationshipInspector):
    def __init__(self, dummy_data: list[Relationship], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dummy_data = dummy_data

    def _build_relationship(self) -> list[Relationship]:
        return self.dummy_data


def test_from_dataloader():
    inspector = MockInspector(dummy_data=[])
    combiner = MetadataCombiner.from_dataloader(
        dataloaders=[],
        max_chunk=10,
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=inspector,
        relationships_inspector_kwargs={},
        relationships=[],
    )


def test_from_dataframe():
    inspector = MockInspector(dummy_data=[])
    combiner = MetadataCombiner.from_dataframe(
        dataframes=[],
        names=[],
        metadata_from_dataloader_kwargs={},
        relationshipe_inspector=inspector,
        relationships_inspector_kwargs={},
        relationships=[],
    )


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
