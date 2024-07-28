import numpy as np
import pandas as pd
import pytest

from sdgx.data_processors.transformers.outlier import OutlierTransformer
from sdgx.data_models.metadata import Metadata


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


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
        df.iat[idx, 2] = "two"  # Introduce string in int column
        df.iat[idx, 3] = "pi"  # Introduce string in float column

    yield df


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

    # Initialize the OutlierTransformer.
    outlier_transformer = OutlierTransformer()
    # Check if the transformer has not been fitted yet.
    assert outlier_transformer.fitted is False

    # Fit the transformer with the DataFrame.
    metadata = Metadata.from_dataframe(outlier_test_df)
    outlier_transformer.fit(metadata=metadata)
    # Check if the transformer has been fitted after the fit operation.
    assert outlier_transformer.fitted

    # Transform the DataFrame using the transformer.
    transformed_df = outlier_transformer.convert(outlier_test_df)

    # Check if the transformed DataFrame does not contain any outliers.
    assert not transformed_df["int_random"].apply(lambda x: isinstance(x, str)).any()
    assert not transformed_df["float_random"].apply(lambda x: isinstance(x, str)).any()

    # Check if the outliers have been replaced with the specified fill values.
    assert transformed_df["int_random"].apply(lambda x: x == 0).sum() > 0
    assert transformed_df["float_random"].apply(lambda x: x == 0).sum() > 0
