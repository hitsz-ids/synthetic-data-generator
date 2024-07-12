import pandas as pd
import pytest

from sdgx.data_models.inspectors.const import ConstInspector


@pytest.fixture
def inspector():
    yield ConstInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def test_const_data(raw_data: pd.DataFrame):
    # Convert the columns to float to allow None values
    raw_data["age"] = raw_data["age"].astype(float)
    raw_data["fnlwgt"] = raw_data["fnlwgt"].astype(float)

    # Set the values to None
    raw_data["age"].values[:] = 100
    raw_data["fnlwgt"].values[:] = 3.14
    raw_data["workclass"].values[:] = "President"

    yield raw_data


def test_inspector(inspector: ConstInspector, test_const_data):
    inspector.fit(test_const_data)
    assert inspector.ready
    assert inspector.const_columns
    assert sorted(inspector.inspect()["const_columns"]) == sorted(["age", "fnlwgt", "workclass"])

    assert inspector.inspect_level == 80
    assert sorted(list(inspector.const_values.keys())) == sorted(["age", "fnlwgt", "workclass"])
    assert inspector.const_values["age"] == 100
    assert inspector.const_values["fnlwgt"] == 3.14
    assert inspector.const_values["workclass"] == "President"


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
