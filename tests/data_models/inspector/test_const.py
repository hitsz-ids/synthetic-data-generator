import pandas as pd
import pytest
import copy

from sdgx.data_models.inspectors.const import ConstInspector


@pytest.fixture
def inspector():
    yield ConstInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def test_const_data(raw_data: pd.DataFrame):
    const_col_df = copy.deepcopy(raw_data)
    
    # Convert the columns to float to allow None values
    const_col_df["age"] = const_col_df["age"].astype(float)
    const_col_df["fnlwgt"] = const_col_df["fnlwgt"].astype(float)

    # Set the values to None
    const_col_df["age"].values[:] = 100
    const_col_df["fnlwgt"].values[:] = 3.14
    const_col_df["workclass"].values[:] = "President"

    yield const_col_df


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
