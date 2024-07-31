import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.nan import NonValueTransformer


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def nan_test_df():
    row_cnt = 1000
    header = ["int_id", "str_id", "int_random", "bool_random"]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    bool_random = int_random < 50

    X = [[int_id[i], str_id[i], int_random[i], bool_random[i]] for i in range(row_cnt)]

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)

    # Randomly add NaN values, 10% of the rows will have NaNs
    nan_indices = np.random.choice(row_cnt, size=int(row_cnt * 0.1), replace=False)
    for idx in nan_indices:
        # Randomly select a column to set to NaN
        col_idx = np.random.randint(0, len(header))
        df.iat[idx, col_idx] = np.nan

    yield df


def has_nan(df):
    """
    This function checks if there are any NaN values in the given DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame to check for NaN values.

    Returns:
    bool: True if there is any NaN value in the DataFrame, False otherwise.
    """
    return df.isnull().values.any()


@pytest.mark.skip(reason="success in local, failed in GitHub Action")
def test_nan_handling_test_df(nan_test_df: pd.DataFrame):
    """
    Test the handling of NaN values in a DataFrame.
    This function tests the behavior of a DataFrame when it contains NaN values.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    nan_test_df (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle NaN values as expected.
    """

    # Check if the DataFrame contains NaN values. If it does, the test will fail.
    assert has_nan(nan_test_df), "NaN values were not removed from the DataFrame."

    # Initialize the NonValueTransformer.
    nan_transformer = NonValueTransformer()
    # Check if the transformer has not been fitted yet.
    assert nan_transformer.fitted is False

    nan_csv_metadata = Metadata.from_dataframe(nan_test_df)
    nan_csv_metadata.column_list = ["int_id", "str_id", "int_random", "bool_random"]

    # Fit the transformer with the DataFrame.
    nan_transformer.fit(nan_csv_metadata)
    # Check if the transformer has been fitted after the fit operation.
    assert nan_transformer.fitted

    # Transform the DataFrame using the transformer.
    transformed_df = nan_transformer.convert(nan_test_df)

    # Check if the transformed DataFrame does not contain any NaN values.
    assert not has_nan(transformed_df)
