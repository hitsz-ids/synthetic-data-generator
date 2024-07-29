import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.empty import EmptyTransformer


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


def test_empty_handling_test_df(test_empty_data: pd.DataFrame):
    """
    Test the handling of empty columns in a DataFrame.
    This function tests the behavior of a DataFrame when it contains empty columns.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    test_empty_data (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle empty columns as expected.
    """

    metadata = Metadata.from_dataframe(test_empty_data)

    # Initialize the EmptyTransformer.
    empty_transformer = EmptyTransformer()
    # Check if the transformer has not been fitted yet.
    assert empty_transformer.fitted is False

    # Fit the transformer with the DataFrame.
    empty_transformer.fit(metadata)

    # Check if the transformer has been fitted after the fit operation.
    assert empty_transformer.fitted

    # Check the empty column
    assert sorted(empty_transformer.empty_columns) == ["age", "fnlwgt"]

    # Transform the DataFrame using the transformer.
    transformed_df = empty_transformer.convert(test_empty_data)

    # Check if the transformed DataFrame does not contain any empty columns.
    # assert not df_has_empty_col(transformed_df)
    processed_metadata = Metadata.from_dataframe(transformed_df)
    assert not processed_metadata.get("empty_columns")

    # reverse convert the df
    reverse_converted_df = empty_transformer.reverse_convert(transformed_df)
    reverse_converted_metadata = Metadata.from_dataframe(reverse_converted_df)
    assert reverse_converted_metadata.get("empty_columns") == {"age", "fnlwgt"}
