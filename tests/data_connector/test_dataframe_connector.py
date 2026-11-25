import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sdgx.data_connectors.dataframe_connector import DataFrameConnector


@pytest.fixture
def data_for_test():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_dataframe_connector(data_for_test):
    df = data_for_test.copy()
    c = DataFrameConnector(data_for_test)
    assert c._columns() == ["a", "b"]
    assert_frame_equal(c._read(), df)
    assert_frame_equal(c._read(offset=1), df[1:])
    assert_frame_equal(c._read(offset=2), df[2:])
    assert c._read(offset=3) is None

    assert_frame_equal(c._read(offset=0), df)
    assert c._read(offset=5555) is None

    # iter
    for d, g in zip(c.iter(chunksize=3), [data_for_test]):
        assert_frame_equal(d, g)

    for d, g in zip(c.iter(chunksize=2), [data_for_test[:2], data_for_test[2:]]):
        assert_frame_equal(d, g)


@pytest.mark.parametrize("offset", [0, 1, 3, 5555])
@pytest.mark.parametrize("empty", [False, True])
def test_read_zero_limit_preserves_schema(offset, empty):
    df = pd.DataFrame(
        {
            "number": pd.Series([1, None, 3], dtype="Int64"),
            "label": pd.Series(["a", "b", "a"], dtype="category"),
        },
    )
    if empty:
        df = df.iloc[:0]
    original = df.copy(deep=True)
    connector = DataFrameConnector(df)

    assert_frame_equal(connector.read(offset=offset, limit=0), df.iloc[:0])
    assert_frame_equal(df, original)


@pytest.mark.parametrize("limit", [None, 1, 10])
@pytest.mark.parametrize("offset", [0, 1, 3])
def test_read_nonzero_limits(data_for_test, offset, limit):
    connector = DataFrameConnector(data_for_test)
    result = connector.read(offset=offset, limit=limit)
    if offset >= len(data_for_test):
        assert result is None
    else:
        stop = None if limit is None else offset + limit
        assert_frame_equal(result, data_for_test.iloc[offset:stop])


def test_empty_dataframe_default_read(data_for_test):
    assert DataFrameConnector(data_for_test.iloc[:0]).read() is None


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
