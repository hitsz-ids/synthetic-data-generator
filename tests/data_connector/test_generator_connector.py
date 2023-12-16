import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sdgx.data_connectors.generator_connector import GeneratorConnector


def generator_caller():
    yield pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    yield pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
    yield pd.DataFrame({"a": [13, 14, 15], "b": [16, 17, 18]})


def test_generator_connector():
    c = GeneratorConnector(generator_caller)
    assert c._columns() == ["a", "b"]
    assert_frame_equal(c._read(), pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert_frame_equal(c._read(offset=1), pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}))
    assert_frame_equal(c._read(offset=2), pd.DataFrame({"a": [13, 14, 15], "b": [16, 17, 18]}))
    assert c._read(offset=3) is None

    # Reset
    assert_frame_equal(c._read(offset=0), pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert_frame_equal(c._read(offset=1000), pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]}))
    assert_frame_equal(c._read(offset=3), pd.DataFrame({"a": [13, 14, 15], "b": [16, 17, 18]}))
    assert c._read(offset=5555) is None

    # iter
    for d, g in zip(c.iter(), generator_caller()):
        assert_frame_equal(d, g)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
