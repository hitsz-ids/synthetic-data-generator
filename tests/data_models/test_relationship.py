from __future__ import annotations

import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import KeyTuple, Relationship
from sdgx.exceptions import RelationshipInitError

parent_metadata = Metadata(
    primary_keys=["parent_id"], column_list=["parent_id"], id_columns={"parent_id"}
)
error_parent_metadata = Metadata(
    primary_keys=["parent_id"], column_list=["parent_id"], id_columns={"foo"}
)
child_metadata = Metadata(
    primary_keys=["child_id"],
    column_list=["parent_id", "child_id"],
    id_columns={"parent_id", "child_id"},
)
error_child_metadata = Metadata(
    primary_keys=["child_id"], column_list=["parent_id", "child_id"], id_columns={"child_id"}
)


@pytest.mark.parametrize(
    "parent_table, parent_metadata, child_table, child_metadata, foreign_keys, exception",
    [
        (
            "parent",
            parent_metadata,
            "child",
            child_metadata,
            [KeyTuple("parent_id", "parent_id")],
            None,
        ),
        (
            "parent",
            error_parent_metadata,
            "child",
            child_metadata,
            [KeyTuple("parent_id", "parent_id")],
            RelationshipInitError,
        ),
        (
            "parent",
            parent_metadata,
            "child",
            error_child_metadata,
            [KeyTuple("parent_id", "parent_id")],
            RelationshipInitError,
        ),
        (
            "parent",
            parent_metadata,
            "parent",
            parent_metadata,
            [KeyTuple("parent_id", "parent_id")],
            RelationshipInitError,
        ),
        ("parent", parent_metadata, "parent", parent_metadata, [], RelationshipInitError),
        (
            "",
            parent_metadata,
            "child",
            child_metadata,
            [KeyTuple("parent_id", "parent_id")],
            RelationshipInitError,
        ),
        (
            "parent",
            parent_metadata,
            "",
            child_metadata,
            [KeyTuple("parent_id", "parent_id")],
            RelationshipInitError,
        ),
        (
            "",
            parent_metadata,
            "",
            child_metadata,
            [KeyTuple("parent_id", "parent_id")],
            RelationshipInitError,
        ),
        ("", parent_metadata, "", child_metadata, [], RelationshipInitError),
    ],
)
def test_build(parent_table, parent_metadata, child_table, child_metadata, foreign_keys, exception):
    if exception:
        with pytest.raises(exception):
            Relationship.build(
                parent_table=parent_table,
                parent_metadata=parent_metadata,
                child_table=child_table,
                child_metadata=child_metadata,
                foreign_keys=foreign_keys,
            )
    else:
        relationship = Relationship.build(
            parent_table=parent_table,
            parent_metadata=parent_metadata,
            child_table=child_table,
            child_metadata=child_metadata,
            foreign_keys=foreign_keys,
        )

        assert relationship.parent_table == parent_table
        assert relationship.child_table == child_table
        assert relationship.foreign_keys == foreign_keys


def test_save_and_load(tmpdir):
    save_file = tmpdir / "relationship.json"
    relationship = Relationship.build(
        parent_table="parent",
        parent_metadata=Metadata(
            primary_keys=["parent_id"], column_list=["parent_id"], id_columns={"parent_id"}
        ),
        child_table="child",
        child_metadata=Metadata(
            primary_keys=["child_id"],
            column_list=["parent_id", "child_id"],
            id_columns={"parent_id", "child_id"},
        ),
        foreign_keys=[KeyTuple("parent_id", "parent_id")],
    )

    relationship.save(save_file)
    assert save_file.exists()

    assert relationship == Relationship.load(save_file)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
