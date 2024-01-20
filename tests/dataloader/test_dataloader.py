from __future__ import annotations

import pandas as pd
import pytest

from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_loader import DataLoader
from sdgx.exceptions import DataLoaderInitError


@pytest.mark.parametrize("cacher", ["NoCache", "DiskCache"])
def test_demo_dataloader(dataloader_builder: DataLoader, cacher, demo_single_table_data_connector):
    d: DataLoader = dataloader_builder(
        data_connector=demo_single_table_data_connector,
        cacher=cacher,
    )
    assert len(d) == 48842
    assert (
        sorted(d.columns())
        == sorted(d.keys())
        == sorted(
            [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "educational-num",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "gender",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "native-country",
                "income",
            ]
        )
    )
    assert d.shape == (48842, 15)
    assert d.load_all().shape == (48842, 15)

    assert d[:].shape == d.shape
    assert d[:100].shape == (100, 15)
    assert d[100:].shape == (48842 - 100, 15)
    assert d[100:10000].shape == (10000 - 100, 15)
    assert d[100:10000:2].shape == ((10000 - 100) // 2, 15)

    assert d[["age", "workclass"]].shape == (48842, 2)

    for df in d.iter():
        assert len(df) == d.chunksize
        break


def generator_caller():
    yield pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    yield pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
    yield pd.DataFrame({"a": [13, 14, 15], "b": [16, 17, 18]})


@pytest.fixture
def generator_connector():
    yield GeneratorConnector(generator_caller)


@pytest.mark.parametrize("cacher", ["NoCache", "DiskCache"])
def test_loader_with_generator_connector(dataloader_builder, cacher, generator_connector):
    if cacher == "NoCache":
        with pytest.raises(DataLoaderInitError):
            d: DataLoader = dataloader_builder(
                data_connector=generator_connector,
                cacher=cacher,
            )
        return
    d: DataLoader = dataloader_builder(
        data_connector=generator_connector,
        cacher=cacher,
    )
    df_all = pd.concat(generator_caller(), ignore_index=True)
    pd.testing.assert_frame_equal(d.load_all(), df_all)
    pd.testing.assert_frame_equal(d[:], df_all[:])
    pd.testing.assert_frame_equal(d[1:], df_all[1:])
    pd.testing.assert_frame_equal(d[:3], df_all[:3])
    pd.testing.assert_frame_equal(d[["a"]], df_all[["a"]])


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
