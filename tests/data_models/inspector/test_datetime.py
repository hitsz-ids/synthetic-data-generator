import datetime

import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.inspectors.datetime import DatetimeInspector


@pytest.fixture
def inspector():
    yield DatetimeInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


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

    h = np.random.randint(0, 24, size=row_cnt)
    m = np.random.randint(0, 59, size=row_cnt)
    s = np.random.randint(0, 59, size=row_cnt)
    date_with_time = [
        simple_datetime[i] + pd.Timedelta(hours=h[i], minutes=m[i], seconds=s[i])
        for i in range(row_cnt)
    ]

    X = [
        [
            int_id[i],
            str_id[i],
            not_int_id[i],
            not_str_id[i],
            simple_datetime[i],
            simple_datetime_2[i],
            date_with_time[i],
        ]
        for i in range(row_cnt)
    ]

    yield pd.DataFrame(X, columns=header)


def test_inspector_demo_data(inspector: DatetimeInspector, raw_data):
    inspector.fit(raw_data)
    assert inspector.ready
    # should be empty set
    assert not inspector.datetime_columns
    assert sorted(inspector.inspect()["datetime_columns"]) == sorted([])


def test_inspector_generated_data(inspector: DatetimeInspector, datetime_test_df: pd.DataFrame):
    # use generated id data
    inspector.fit(datetime_test_df)
    assert inspector.datetime_columns
    assert sorted(inspector.inspect()["datetime_columns"]) == sorted(
        ["simple_datetime", "simple_datetime_2", "date_with_time"]
    )


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
