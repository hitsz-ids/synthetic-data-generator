import pandas as pd
import pytest

from sdgx.data_models.inspectors.subset_relationship import SubsetRelationshipInspector
from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship


@pytest.fixture
def inspector():
    yield SubsetRelationshipInspector()


@pytest.fixture
def dummy_data(demo_relational_table_path):
    table_path_a, table_path_b, _ = demo_relational_table_path
    df_a = pd.read_csv(table_path_a)
    df_b = pd.read_csv(table_path_b)

    yield [
        (df_a, "parent", Metadata.from_dataframe(df_a)),
        (df_b, "child", Metadata.from_dataframe(df_b)),
    ]


@pytest.fixture
def dummy_relationship(demo_relational_table_path):
    _, _, pairs = demo_relational_table_path

    yield Relationship.build(
        parent_table="parent",
        child_table="child",
        foreign_keys=pairs,
    )


def test_inspector(dummy_data, dummy_relationship, inspector: SubsetRelationshipInspector):
    for raw_data, name, metadata in dummy_data:
        inspector.fit(raw_data, name=name, metadata=metadata)
    relationships = inspector.inspect()["relationships"]
    assert relationships
    assert relationships == [dummy_relationship]


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
