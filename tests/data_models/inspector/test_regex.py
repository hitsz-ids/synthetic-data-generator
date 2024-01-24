import datetime

import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.inspectors.regex import RegexInspector


@pytest.fixture
def int_inspector():
    yield RegexInspector(
        pattern= "^[0-9]*$",
        data_type_name= "int"
    )

@pytest.fixture
def empty_inspector():
    yield RegexInspector(
        pattern= "^$",
        data_type_name= "empty"
    )

@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


def test_int_regex_inspector_demo_data(int_inspector: RegexInspector,
                                   raw_data: pd.DataFrame):
    int_inspector.fit(raw_data)
    assert int_inspector.ready
    # should not be empty set
    assert int_inspector.regex_columns
    assert sorted(int_inspector.inspect()["int_columns"]) == sorted(['age', 'capitalgain', 'capitalloss', 'education-num', 'fnlwgt', 'hoursperweek'])
    assert int_inspector.inspect_level == 10

def test_empty_regex_inspector_demo_data(empty_inspector: RegexInspector,
                                   raw_data: pd.DataFrame):
    empty_inspector.fit(raw_data)
    assert empty_inspector.ready
    # should be empty set
    assert not empty_inspector.regex_columns
    assert sorted(empty_inspector.inspect()["empty_columns"]) == sorted([])
    assert empty_inspector.inspect_level == 10

if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
