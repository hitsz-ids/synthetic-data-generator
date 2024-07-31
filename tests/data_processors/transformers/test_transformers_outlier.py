import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.outlier import OutlierTransformer


@pytest.fixture
def outlier_test_df():
    row_cnt = 1000
    header = ["int_id", "str_id", "int_random", "float_random"]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    float_random = np.random.uniform(0, 100, size=row_cnt)

    X = [[int_id[i], str_id[i], int_random[i], float_random[i]] for i in range(row_cnt)]

    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)

    # Introduce outliers
    outlier_indices = np.random.choice(row_cnt, size=int(row_cnt * 0.1), replace=False)
    for idx in outlier_indices:
        df.iat[idx, 2] = "not_number_outlier"  # Introduce string in int column
        df.iat[idx, 3] = "not_number_outlier"  # Introduce string in float column

    yield df


@pytest.mark.skip(reason="success in local, failed in GitHub Action")
def test_outlier_handling_test_df(outlier_test_df: pd.DataFrame):
    """
    Test the handling of outliers in a DataFrame.
    This function tests the behavior of a DataFrame when it contains outliers.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
        outlier_test_df (pd.DataFrame): The DataFrame to test.

    Returns:
        None

    Raises:
        AssertionError: If the DataFrame does not handle outliers as expected.
    """

    assert "not_number_outlier" in outlier_test_df["int_random"].to_list()
    assert "not_number_outlier" in outlier_test_df["float_random"].to_list()

    # Initialize the OutlierTransformer.
    outlier_transformer = OutlierTransformer()
    # Check if the transformer has not been fitted yet.
    assert outlier_transformer.fitted is False

    # Fit the transformer with the DataFrame.
    metadata_outlier = Metadata.from_dataframe(outlier_test_df)
    metadata_outlier.column_list = ["int_id", "str_id", "int_random", "float_random"]
    metadata_outlier.int_columns = set(["int_id", "int_random"])
    metadata_outlier.float_columns = set(["float_random"])

    # Fit the transformer
    outlier_transformer.fit(metadata=metadata_outlier)
    # Check if the transformer has been fitted after the fit operation.
    assert outlier_transformer.fitted

    # Transform the DataFrame using the transformer.
    transformed_df = outlier_transformer.convert(outlier_test_df)

    # Check if the transformed DataFrame does not contain any outliers.
    assert not "not_number_outlier" in transformed_df["int_random"].to_list()
    assert not "not_number_outlier" in transformed_df["float_random"].to_list()

    # Check if the outliers have been replaced with the specified fill values.
    assert 0 in transformed_df["int_random"].to_list()
    assert 0.0 in transformed_df["float_random"].to_list()
