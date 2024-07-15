import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.const import ConstValueTransformer


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
    raw_data["fnlwgt"].values[:] = 1.41421
    raw_data["workclass"].values[:] = "President"

    yield raw_data


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
    assert sorted(const_transformer.const_columns) == ["age", "fnlwgt", 'workclass',]

    # Transform the DataFrame using the transformer.
    transformed_df = const_transformer.convert(test_const_data)

    # Check if the transformed DataFrame does not contain any const columns.
    # assert not df_has_const_col(transformed_df)
    processed_metadata = Metadata.from_dataframe(transformed_df)
    assert not processed_metadata.get("const_columns")

    # reverse convert the df
    reverse_converted_df = const_transformer.reverse_convert(transformed_df)
    reverse_converted_metadata = Metadata.from_dataframe(reverse_converted_df)
    assert reverse_converted_metadata.get("const_columns") == {"age", "fnlwgt", 'workclass'}
    assert reverse_converted_df["age"][0] == 100
    assert reverse_converted_df["fnlwgt"][0] == 1.41421
    assert reverse_converted_df["fnlwgt"][0] == "President"
