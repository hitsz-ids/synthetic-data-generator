from __future__ import annotations

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
    assert metadata.int_columns == metadata.get("int_columns")
    assert metadata.float_columns == metadata.get("float_columns")

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


def test_metadata_primary_query_filed_tags():
    metadata = Metadata()
    metadata.set("id_columns", {"user_id"})
    metadata.set("int_columns", {"user_id", "age"})

    # query the "user_id" field tags
    results = metadata.query("user_id")
    results_list = list(results)
    print(results_list)
    assert "id_columns" in results_list
    assert "int_columns" in results_list


def test_metadata_check(metadata: Metadata):
    # For the example table, it does not contain a primary key
    # but it can be used in  single-table data synthetic tasks.
    # clear primary key
    metadata.update_primary_key([])
    # do meatadata check
    metadata.check()


def test_demo_multi_table_data_metadata_parent(demo_multi_data_parent_matadata):
    # self check
    demo_multi_data_parent_matadata.check()
    # check each col's data type
    assert demo_multi_data_parent_matadata.get_column_data_type("Store") == "id"
    assert demo_multi_data_parent_matadata.get_column_data_type("StoreType") == "discrete"
    assert demo_multi_data_parent_matadata.get_column_data_type("Assortment") == "discrete"
    assert demo_multi_data_parent_matadata.get_column_data_type("CompetitionDistance") == "int"
    assert (
        demo_multi_data_parent_matadata.get_column_data_type("CompetitionOpenSinceMonth") == "int"
    )
    assert demo_multi_data_parent_matadata.get_column_data_type("Promo2") == "int"
    assert demo_multi_data_parent_matadata.get_column_data_type("Promo2SinceWeek") == "int"
    assert demo_multi_data_parent_matadata.get_column_data_type("Promo2SinceYear") == "int"
    assert demo_multi_data_parent_matadata.get_column_data_type("PromoInterval") == "discrete"
    # check pii
    for each_col in demo_multi_data_parent_matadata.column_list:
        assert demo_multi_data_parent_matadata.get_column_pii(each_col) is False
    assert len(demo_multi_data_parent_matadata.pii_columns) is 0
    # check dump
    assert "column_data_type" in demo_multi_data_parent_matadata.dump().keys()


def test_demo_multi_table_data_metadata_child(demo_multi_data_child_matadata):
    # self check
    demo_multi_data_child_matadata.check()
    # check each col's data type
    assert demo_multi_data_child_matadata.get_column_data_type("Store") == "int"
    assert demo_multi_data_child_matadata.get_column_data_type("Date") == "datetime"
    assert demo_multi_data_child_matadata.get_column_data_type("Customers") == "int"
    assert demo_multi_data_child_matadata.get_column_data_type("StateHoliday") == "const"
    assert demo_multi_data_child_matadata.get_column_data_type("Sales") == "int"
    assert demo_multi_data_child_matadata.get_column_data_type("Promo") == "int"
    assert demo_multi_data_child_matadata.get_column_data_type("DayOfWeek") == "int"
    assert demo_multi_data_child_matadata.get_column_data_type("Open") == "int"
    assert demo_multi_data_child_matadata.get_column_data_type("SchoolHoliday") == "int"
    # check pii
    for each_col in demo_multi_data_child_matadata.column_list:
        assert demo_multi_data_child_matadata.get_column_pii(each_col) is False
    assert len(demo_multi_data_child_matadata.pii_columns) is 0
    # check dump
    assert "column_data_type" in demo_multi_data_child_matadata.dump().keys()


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
