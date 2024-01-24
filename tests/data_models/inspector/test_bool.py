import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.inspectors.bool import BoolInspector
from sdgx.exceptions import InspectorInitError


@pytest.fixture
def inspector():
    yield BoolInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def bool_test_df():
    row_cnt = 1000
    header = ["int_id", "str_id", "int_random", "bool_random"]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    bool_random = int_random < 5

    X = [[int_id[i], str_id[i], int_random[i], bool_random[i]] for i in range(row_cnt)]

    yield pd.DataFrame(X, columns=header)


def test_inspector_demo_data(inspector: BoolInspector, raw_data):
    inspector.fit(raw_data)
    assert inspector.ready
    # should be empty set
    assert not inspector.bool_columns
    assert sorted(inspector.inspect()["bool_columns"]) == sorted([])
    assert inspector.inspect_level == 10


def test_inspector_generated_data(inspector: BoolInspector, bool_test_df: pd.DataFrame):
    # use generated id data
    inspector.fit(bool_test_df)
    assert inspector.bool_columns
    assert sorted(inspector.inspect()["bool_columns"]) == sorted(["bool_random"])


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
