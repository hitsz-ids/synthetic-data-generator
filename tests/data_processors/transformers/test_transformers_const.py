import copy

import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.const import ConstValueTransformer


@pytest.fixture
def test_const_data(demo_single_table_path):

    const_col_df = pd.read_csv(demo_single_table_path)
    # Convert the columns to float to allow None values
    const_col_df["age"] = const_col_df["age"].astype(float)
    const_col_df["fnlwgt"] = const_col_df["fnlwgt"].astype(float)

    # Set the values to None
    const_col_df["age"].values[:] = 100
    const_col_df["fnlwgt"].values[:] = 1.41421
    const_col_df["workclass"].values[:] = "President"

    yield const_col_df


def test_const_handling_test_df(test_const_data: pd.DataFrame):
    """
    Test the handling of const columns in a DataFrame.
    This function tests the behavior of a DataFrame when it contains const columns.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    test_const_data (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle const columns as expected.
    """

    metadata = Metadata.from_dataframe(test_const_data)

    # Initialize the ConstValueTransformer.
    const_transformer = ConstValueTransformer()
    # Check if the transformer has not been fitted yet.
    assert const_transformer.fitted is False

    # Fit the transformer with the DataFrame.
    const_transformer.fit(metadata)

    # Check if the transformer has been fitted after the fit operation.
    assert const_transformer.fitted

    # Check the const column
    assert sorted(const_transformer.const_columns) == [
        "age",
        "fnlwgt",
        "workclass",
    ]

    # Transform the DataFrame using the transformer.
    transformed_df = const_transformer.convert(test_const_data)

    assert "age" not in transformed_df.columns
    assert "fnlwgt" not in transformed_df.columns
    assert "workclass" not in transformed_df.columns

    # reverse convert the df
    reverse_converted_df = const_transformer.reverse_convert(transformed_df)

    assert "age" in reverse_converted_df.columns
    assert "fnlwgt" in reverse_converted_df.columns
    assert "workclass" in reverse_converted_df.columns

    assert reverse_converted_df["age"][0] == 100
    assert reverse_converted_df["fnlwgt"][0] == 1.41421
    assert reverse_converted_df["workclass"][0] == "President"

    assert len(reverse_converted_df["age"].unique()) == 1
    assert len(reverse_converted_df["fnlwgt"].unique()) == 1
    assert len(reverse_converted_df["workclass"].unique()) == 1
