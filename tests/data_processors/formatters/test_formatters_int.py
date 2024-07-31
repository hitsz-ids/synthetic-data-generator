import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.formatters.int import IntValueFormatter


def int_formatter_df():
    row_cnt = 1000
    header = ["int_id", "str_id", "int_random", "float_random"]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    float_random = np.random.randn(row_cnt)

    X = [[int_id[i], str_id[i], int_random[i], float_random[i]] for i in range(row_cnt)]
    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)
    return df


def is_an_integer_list(lst):
    """
    Check if all elements in the list are integers or floats that are also integers.

    Parameters:
    lst (list): The list to be checked.

    Returns:
    bool: True if all elements are integers or floats that are also integers, False otherwise.
    """
    return all(isinstance(i, int) or (isinstance(i, float) and i.is_integer()) for i in lst)


def test_int_formatter_fit_test_df():
    """
    Test the functionality of the IntValueFormatter class.

    This function tests the following:
    1. The fit method of the IntValueFormatter class.
    2. The addition of a new column to the formatter.
    3. The reverse conversion of the DataFrame.
    4. The checking of integer values in the DataFrame columns.

    Parameters:
    df_data (pd.DataFrame): The DataFrame to be tested.

    Returns:
    None

    Raises:
    AssertionError: If any of the assertions fail.
    """
    df = int_formatter_df()
    # get metadata
    metadata_df = Metadata.from_dataframe(df)

    # fit the formatter
    formatter = IntValueFormatter()
    formatter.fit(metadata_df)
    assert sorted(metadata_df.column_list) == sorted(["int_id", "str_id", "int_random", "float_random"])
    assert formatter.int_columns == {"int_random", "int_id"}
    # add float_random column to formatter
    formatter.int_columns.add("float_random")
    assert formatter.int_columns == {"int_random", "int_id", "float_random"}
    reverse_df = formatter.reverse_convert(df)
    assert is_an_integer_list(reverse_df["float_random"].tolist())
    assert is_an_integer_list(reverse_df["int_id"].tolist())
    assert not is_an_integer_list(reverse_df["str_id"].tolist())
    assert is_an_integer_list(reverse_df["int_random"].tolist())
