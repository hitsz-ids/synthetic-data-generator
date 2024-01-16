from __future__ import annotations

import pytest

from sdgx.data_models.relationship import KeyTuple, Relationship
from sdgx.exceptions import RelationshipInitError


@pytest.mark.parametrize(
    "parent_table, child_table, foreign_keys, exception",
    [
        ("parent", "child", [KeyTuple("parent_id", "parent_id")], None),
        (
            "parent",
            "child",
            [KeyTuple("parent_id", "parent_id"), KeyTuple("child_id", "child_id")],
            None,
        ),
        ("parent", "child", [KeyTuple("parent_id", "p_id_in_child")], None),
        ("parent", "parent", [KeyTuple("parent_id", "parent_id")], RelationshipInitError),
        ("parent", "parent", [], RelationshipInitError),
        ("", "child", [KeyTuple("parent_id", "parent_id")], RelationshipInitError),
        ("parent", "", [KeyTuple("parent_id", "parent_id")], RelationshipInitError),
        ("", "", [KeyTuple("parent_id", "parent_id")], RelationshipInitError),
        ("", "", [], RelationshipInitError),
    ],
)
def test_build(parent_table, child_table, foreign_keys, exception):
    if exception:
        with pytest.raises(exception):
            Relationship.build(
                parent_table=parent_table,
                child_table=child_table,
                foreign_keys=foreign_keys,
            )
    else:
        relationship = Relationship.build(
            parent_table=parent_table,
            child_table=child_table,
            foreign_keys=foreign_keys,
        )

        assert relationship.parent_table == parent_table
        assert relationship.child_table == child_table
        assert relationship.foreign_keys == foreign_keys


def test_save_and_load(tmpdir):
    save_file = tmpdir / "relationship.json"
    relationship = Relationship.build(
        parent_table="parent",
        child_table="child",
        foreign_keys=["parent_id"],
    )

    relationship.save(save_file)
    assert save_file.exists()

    assert relationship == Relationship.load(save_file)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
