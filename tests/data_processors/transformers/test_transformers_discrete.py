import random

import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.column_order import ColumnOrderTransformer
from sdgx.data_processors.transformers.discrete import DiscreteTransformer


@pytest.fixture
def df_data():
    row_cnt = 1000
    header = ["int_id", "discrete_val", "int_random", "bool_random", "float_random"]

    int_id = list(range(row_cnt))
    discrete_val = list(random.choice(["a", "b", "c"]) for _ in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    bool_random = int_random < 50
    float_random = np.random.randn(row_cnt)

    X = [
        [int_id[i], discrete_val[i], int_random[i], bool_random[i], float_random[i]]
        for i in range(row_cnt)
    ]
    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)
    yield df


def is_a_string_list(lst):
    """
    Check if all items in a list are strings.

    Parameters:
    lst (list): The list to check.

    Returns:
    bool: True if all items in the list are strings, False otherwise.
    """
    return all(isinstance(item, str) for item in lst)


def is_an_integer_list(lst):
    """
    Check if all elements in the list are integers or floats that are also integers.

    Parameters:
    lst (list): The list to be checked.

    Returns:
    bool: True if all elements are integers or floats that are also integers, False otherwise.
    """
    return all(isinstance(i, int) or (isinstance(i, float) and i.is_integer()) for i in lst)


def test_discrete_transformer_fit_test_df(df_data: pd.DataFrame):
    """
    Test the fit and convert methods of the DiscreteTransformer class.

    This function tests the following:
    1. The fit method of the DiscreteTransformer class.
    2. The convert method of the DiscreteTransformer class.
    3. The reverse_convert method of the DiscreteTransformer class.
    4. The equality of the original dataframe and the reversely converted dataframe.

    Parameters:
    df_data (pd.DataFrame): The input dataframe to be tested.

    Returns:
    None
    """
    # get metadata
    metadata_df = Metadata.from_dataframe(df_data)

    # use another ColumnOrderTransformer to reorder the columns
    order_transformer = ColumnOrderTransformer()
    order_transformer.fit(metadata_df)

    # fit the transformer
    transformer = DiscreteTransformer()
    assert not transformer.fitted
    transformer.fit(metadata_df, df_data)
    assert transformer.fitted
    assert transformer.discrete_columns == {"discrete_val"}

    # convert df using the fitted transformer
    converted_df = transformer.convert(df_data)
    assert isinstance(converted_df, pd.DataFrame)
    assert is_an_integer_list(converted_df["discrete_val_a"].to_list())
    assert is_an_integer_list(converted_df["discrete_val_b"].to_list())
    assert is_an_integer_list(converted_df["discrete_val_c"].to_list())

    # reverse convert the df back to original
    reverse_converted_df = transformer.reverse_convert(converted_df)
    reverse_converted_df = order_transformer.reverse_convert(reverse_converted_df)
    assert isinstance(reverse_converted_df, pd.DataFrame)
    assert is_a_string_list(reverse_converted_df["discrete_val"].to_list())

    # check if the dataframe is equal to the original one
    # use the eq method and .all().all() to check the equality of two 2d dataframes
    assert reverse_converted_df.eq(df_data).all().all()
