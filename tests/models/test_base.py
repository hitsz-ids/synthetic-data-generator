from __future__ import annotations

from collections import defaultdict, namedtuple

import pytest

from sdgx.models.statistics.multi_tables.base import MultiTableSynthesizerModel


@pytest.fixture
def demo_base_multi_table_synthesizer(
    demo_multi_table_data_metadata_combiner, demo_multi_table_data_loader
):
    yield MultiTableSynthesizerModel(
        use_dataloader=True,
        metadata_combiner=demo_multi_table_data_metadata_combiner,
        tables_data_loader=demo_multi_table_data_loader,
    )


def test_base_multi_table_synthesizer(demo_base_multi_table_synthesizer):
    KeyTuple = namedtuple("KeyTuple", ["parent", "child"])

    assert demo_base_multi_table_synthesizer.parent_map == defaultdict(None, {"train": "store"})
    assert demo_base_multi_table_synthesizer.child_map == defaultdict(None, {"store": "train"})
    assert demo_base_multi_table_synthesizer._get_all_foreign_keys("train")[0][0] == KeyTuple(
        parent="Store", child="Store"
    )


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
