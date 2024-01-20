import pandas as pd
import pytest

from sdgx.data_models.inspectors.i_id import IDInspector


@pytest.fixture
def inspector():
    yield IDInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def id_test_df():
    row_cnt = 1000
    header = ["int_id", "str_id", "not_int_id", "not_str_id"]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    not_int_id = list(range(int(row_cnt / 2))) + list(range(int(row_cnt / 2)))
    not_str_id = list("id_" + str(i) for i in range(int(row_cnt / 2))) + list(
        "id_" + str(i) for i in range(int(row_cnt / 2))
    )

    X = [[int_id[i], str_id[i], not_int_id[i], not_str_id[i]] for i in range(row_cnt)]

    yield pd.DataFrame(X, columns=header)


def test_inspector_demo_data(inspector: IDInspector, raw_data):
    inspector.fit(raw_data)
    assert inspector.ready
    # should be empty set
    assert not inspector.ID_columns
    assert sorted(inspector.inspect()["id_columns"]) == sorted([])
    assert inspector.inspect_level == 20


def test_inspector_generated_data(inspector: IDInspector, id_test_df: pd.DataFrame):
    # use generated id data
    inspector.fit(id_test_df)
    assert inspector.ID_columns
    assert sorted(inspector.inspect()["id_columns"]) == sorted(["int_id", "str_id"])


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
