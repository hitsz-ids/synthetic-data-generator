from __future__ import annotations

import pytest

from sdgx.data_loader import DataLoader


def test_dataloader(demo_single_table_data_loader: DataLoader):
    assert len(demo_single_table_data_loader) == 48843
    assert (
        sorted(demo_single_table_data_loader.columns())
        == sorted(demo_single_table_data_loader.keys())
        == sorted(
            [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capitalgain",
                "capitalloss",
                "hoursperweek",
                "native-country",
                "class",
            ]
        )
    )
    assert demo_single_table_data_loader.shape == (48843, 15)
    assert demo_single_table_data_loader.load_all().shape == (48843, 15)

    assert demo_single_table_data_loader[:100].shape == (100, 15)
    assert demo_single_table_data_loader[100:].shape == (48843 - 100, 15)
    assert demo_single_table_data_loader[100:10000].shape == (10000 - 100, 15)
    assert demo_single_table_data_loader[100:10000:2].shape == ((10000 - 100) // 2, 15)

    assert demo_single_table_data_loader[["age", "workclass"]].shape == (48843, 2)

    for df in demo_single_table_data_loader.iter():
        assert len(df) == demo_single_table_data_loader.chunksize
        break


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
