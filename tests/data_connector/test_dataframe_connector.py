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


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
