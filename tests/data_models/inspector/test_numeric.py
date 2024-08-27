import pandas as pd
import pytest

from sdgx.data_models.inspectors.numeric import NumericInspector


@pytest.fixture
def inspector():
    yield NumericInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


def test_inspector(inspector: NumericInspector, raw_data):
    inspector.fit(raw_data)
    assert inspector.ready
    assert inspector.int_columns
    assert sorted(inspector.inspect()["int_columns"]) == sorted(
        ["educational-num", "fnlwgt", "hours-per-week", "age", "capital-gain", "capital-loss"]
    )
    assert not inspector.float_columns
    assert inspector.inspect_level == 10
    assert inspector.negative_columns == set()
    assert inspector.positive_columns == {"age", "hours-per-week", "fnlwgt", "educational-num"}
    assert set(inspector.inspect().keys()) == {"int_columns", "float_columns", "numeric_format"}


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
