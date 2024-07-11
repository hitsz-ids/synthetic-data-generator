import pandas as pd
import pytest

from sdgx.data_models.inspectors.empty import EmptyInspector


@pytest.fixture
def inspector():
    yield EmptyInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def test_empty_data(raw_data: pd.DataFrame):
    # Convert the columns to float to allow None values
    raw_data["age"] = raw_data["age"].astype(float)
    raw_data["fnlwgt"] = raw_data["fnlwgt"].astype(float)

    # Set the values to None
    raw_data["age"].values[:] = None
    raw_data["fnlwgt"].values[:] = None

    yield raw_data


def test_inspector(inspector: EmptyInspector, test_empty_data):
    inspector.fit(test_empty_data)
    assert inspector.ready
    assert inspector.empty_columns
    assert sorted(inspector.inspect()["empty_columns"]) == sorted(
        [
            "age",
            "fnlwgt",
        ]
    )
    assert inspector.inspect_level == 90


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
