import datetime

import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.inspectors.regex import RegexInspector
from sdgx.exceptions import InspectorInitError


@pytest.fixture
def int_inspector():
    yield RegexInspector(pattern="^[0-9]*$", data_type_name="int")


@pytest.fixture
def empty_inspector():
    yield RegexInspector(pattern="^$", data_type_name="empty_columns", match_percentage=0.88)


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


def test_int_regex_inspector_demo_data(int_inspector: RegexInspector, raw_data: pd.DataFrame):
    int_inspector.fit(raw_data)
    assert int_inspector.ready
    # should not be empty set
    assert int_inspector.regex_columns
    assert sorted(int_inspector.inspect()["int_columns"]) == sorted(
        ["age", "capital-gain", "capital-loss", "educational-num", "fnlwgt", "hours-per-week"]
    )
    assert int_inspector.inspect_level == 10


def test_empty_regex_inspector_demo_data(empty_inspector: RegexInspector, raw_data: pd.DataFrame):
    empty_inspector.fit(raw_data)
    assert empty_inspector.ready
    # should be empty set
    assert not empty_inspector.regex_columns
    assert sorted(empty_inspector.inspect()["empty_columns"]) == sorted([])
    assert empty_inspector.inspect_level == 10


def test_match_rate_property(empty_inspector: RegexInspector):
    assert empty_inspector.match_percentage == 0.88
    empty_inspector.match_percentage = 0.7
    empty_inspector.match_percentage = 1

    try:
        empty_inspector.match_percentage = 1.2
    except Exception as e:
        assert type(e) == InspectorInitError

    try:
        empty_inspector.match_percentage = 0.5
    except Exception as e:
        assert type(e) == InspectorInitError
    pass


def test_parameter_missing_case():
    # init only with pattern
    only_pattern_inspector = RegexInspector(pattern="^[0-9]*$")
    assert only_pattern_inspector.data_type_name == "regex_^[0-9]*$_columns"
    # init without pattern
    has_error = False
    try:
        miss_pattern_inspector = RegexInspector(data_type_name="xx")
    except Exception as e:
        has_error = True
        assert type(e) == InspectorInitError
    assert has_error is True
    # init without data type name and pattern
    has_error = False
    try:
        dtype_pattern_inspector = RegexInspector()
    except Exception as e:
        has_error = True
        assert type(e) == InspectorInitError
    assert has_error is True


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
