import datetime

import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.formatters.datetime import DatetimeFormatter


@pytest.fixture
def datetime_test_df():
    row_cnt = 1000
    header = [
        "int_id",
        "str_id",
        "not_int_id",
        "not_str_id",
        "simple_datetime",
        "simple_datetime_2",
        "date_with_time",
    ]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    not_int_id = list(range(int(row_cnt / 2))) + list(range(int(row_cnt / 2)))
    not_str_id = list("id_" + str(i) for i in range(int(row_cnt / 2))) + list(
        "id_" + str(i) for i in range(int(row_cnt / 2))
    )

    simple_datetime = pd.date_range(start="2023-12-27", periods=row_cnt)

    # such "%d %b %Y" cannot be directly marked as a datetime column by pandas
    # but can be identified through the to_datetime method, which is implemented by DatetimeInspector
    simple_datetime_2 = [datetime.datetime.strftime(x, "%d %b %Y") for x in simple_datetime]
    simple_datetime_str = [datetime.datetime.strftime(x, "%Y-%m-%d") for x in simple_datetime]

    h = np.random.randint(0, 24, size=row_cnt)
    m = np.random.randint(0, 59, size=row_cnt)
    s = np.random.randint(0, 59, size=row_cnt)
    date_with_time = [
        simple_datetime[i] + pd.Timedelta(hours=h[i], minutes=m[i], seconds=s[i])
        for i in range(row_cnt)
    ]

    datetime_test_df = [
        [
            int_id[i],
            str_id[i],
            not_int_id[i],
            not_str_id[i],
            simple_datetime_str[i],
            simple_datetime_2[i],
            date_with_time[i],
        ]
        for i in range(row_cnt)
    ]

    yield pd.DataFrame(datetime_test_df, columns=header)  # 1000 rows, 7 columns


def is_an_integer_list(lst):
    """
    Check if all elements in the list are integers or floats that are also integers.

    Parameters:
    lst (list): The list to be checked.

    Returns:
    bool: True if all elements are integers or floats that are also integers, False otherwise.
    """
    return all(isinstance(i, int) or (isinstance(i, float) and i.is_integer()) for i in lst)


def is_a_string_list(lst):
    """
    Check if all items in a list are strings.

    Parameters:
    lst (list): The list to check.

    Returns:
    bool: True if all items in the list are strings, False otherwise.
    """
    return all(isinstance(item, str) for item in lst)


def test_datetime_formatter_test_df_dead_column(datetime_test_df: pd.DataFrame):
    """
    Test the DatetimeFormatter class with a DataFrame that has datetime columns.

    Parameters:
    datetime_test_df (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If any of the assertions fail.
    """
    # about the data
    assert datetime_test_df.shape == (1000, 7)

    # get the metadata
    metadata_df = Metadata.from_dataframe(datetime_test_df)
    assert metadata_df.datetime_columns == {
        "simple_datetime_2",
        "date_with_time",
        "simple_datetime",
    }
    metadata_df.datetime_format = {}

    # fit the transformer
    transformer = DatetimeFormatter()
    transformer.fit(metadata=metadata_df)

    # no element in datetime_columns
    assert transformer.datetime_columns == []
    assert set(transformer.dead_columns) == {
        "simple_datetime_2",
        "date_with_time",
        "simple_datetime",
    }  # all dead


def test_datetime_formatter_test_df(datetime_test_df: pd.DataFrame):
    """
    Test function for the DatetimeFormatter class.

    This function tests the functionality of the DatetimeFormatter class by creating a test DataFrame,
    setting the datetime format for the columns, fitting the transformer, converting the DataFrame,
    reversing the conversion, and checking if the reversed DataFrame is equal to the original one.

    Args:
        datetime_test_df (pd.DataFrame): The test DataFrame to be used for testing.

    Returns:
        None

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # about the data
    assert datetime_test_df.shape == (1000, 7)

    # get the metadata
    metadata_df = Metadata.from_dataframe(datetime_test_df)
    assert metadata_df.datetime_columns == {
        "simple_datetime_2",
        "date_with_time",
        "simple_datetime",
    }
    # set the right format
    # TODO : it seems that metadata has problems in inspecting defalut format such as "%d %b %Y"
    # here we assign the real value, and will fix this issue in another PR
    datetime_format = {}
    datetime_format["simple_datetime"] = "%Y-%m-%d"
    datetime_format["simple_datetime_2"] = "%d %b %Y"
    datetime_format["date_with_time"] = "%Y-%m-%d %H:%M:%S"
    metadata_df.datetime_format = datetime_format

    # fit the transformer
    transformer = DatetimeFormatter()
    assert not transformer.fitted
    transformer.fit(metadata=metadata_df)
    assert transformer.fitted

    # no element in dead_columns
    assert transformer.dead_columns == []  # no columns dead
    assert set(transformer.datetime_columns) == {
        "simple_datetime_2",
        "date_with_time",
        "simple_datetime",
    }

    # convert the dataframe, check if datetime columns are int type
    converted_df = transformer.convert(datetime_test_df)
    assert is_an_integer_list(converted_df["date_with_time"].to_list())
    assert is_an_integer_list(converted_df["simple_datetime_2"].to_list())
    assert is_an_integer_list(converted_df["simple_datetime"].to_list())

    reverse_converte_df = transformer.reverse_convert(converted_df)
    # assert string type in reverse converted dataframe
    assert is_a_string_list(reverse_converte_df["simple_datetime"].to_list())
    assert is_a_string_list(reverse_converte_df["date_with_time"].to_list())
    assert is_a_string_list(reverse_converte_df["simple_datetime_2"].to_list())

    # check if the dataframe is equal to the original one
    # use the eq method and .all().all() to check the equality of two 2d dataframes
    assert reverse_converte_df.eq(datetime_test_df).all().all()
