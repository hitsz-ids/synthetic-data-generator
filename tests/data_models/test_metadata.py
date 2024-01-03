import json
from pathlib import Path

import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata


@pytest.fixture
def dataloader(demo_single_table_path, cacher_kwargs):
    d = DataLoader(CsvConnector(path=demo_single_table_path), cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize(clear_cache=True)


@pytest.fixture
def metadata(dataloader):
    yield Metadata.from_dataloader(dataloader)


def test_metadata(metadata: Metadata):
    assert metadata.discrete_columns == metadata.get("discrete_columns")
    assert metadata.id_columns == metadata.get("id_columns")
    assert metadata.datetime_columns == metadata.get("datetime_columns")
    assert metadata.bool_columns == metadata.get("bool_columns")
    assert metadata.numeric_columns == metadata.get("numeric_columns")

    metadata.set("a", "something")
    assert metadata.get("a") == set(["something"])

    assert metadata._dump_json()


def test_metadata_save_load(metadata: Metadata, tmp_path: Path):
    test_path = tmp_path / "metadata_path_test.json"
    metadata.save(test_path)
    # load from path
    new_meatadata = Metadata.load(test_path)
    assert metadata == new_meatadata


def test_metadata_primary_key(metadata: Metadata):
    # inspect fnlwgt to ID type (for test)
    metadata.add("id_columns", "fnlwgt")
    # set fnlwgt as primary key
    metadata.update_primary_key(["fnlwgt"])
    assert metadata.primary_keys == set(["fnlwgt"])


def test_metadata_check(metadata: Metadata):
    # For the example table, it does not contain a primary key
    # but it can be used in  single-table data synthetic tasks.
    # clear primary key
    metadata.update_primary_key([])
    # do meatadata check
    metadata.check()


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
