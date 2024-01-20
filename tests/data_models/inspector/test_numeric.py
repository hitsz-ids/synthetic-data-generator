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
    assert inspector.numeric_columns
    assert sorted(inspector.inspect()["numeric_columns"]) == sorted(
        ["educational-num", "fnlwgt", "hours-per-week", "age", "capital-gain", "capital-loss"]
    )


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
