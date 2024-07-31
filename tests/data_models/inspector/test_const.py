import copy

import pandas as pd
import pytest

from sdgx.data_models.inspectors.const import ConstInspector


@pytest.fixture
def test_const_data(demo_single_table_path):
    const_col_df = pd.read_csv(demo_single_table_path)

    # Convert the columns to float to allow None values
    const_col_df["age"] = const_col_df["age"].astype(float)
    const_col_df["fnlwgt"] = const_col_df["fnlwgt"].astype(float)

    # Set the values to None
    const_col_df["age"].values[:] = 100
    const_col_df["fnlwgt"].values[:] = 3.14
    const_col_df["workclass"].values[:] = "President"

    yield const_col_df


def test_const_inspector(test_const_data: pd.DataFrame):
    inspector = ConstInspector()
    inspector.fit(test_const_data)
    assert inspector.ready
    assert inspector.const_columns

    assert sorted(inspector.inspect()["const_columns"]) == sorted(["age", "fnlwgt", "workclass"])
    assert inspector.inspect_level == 80


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
