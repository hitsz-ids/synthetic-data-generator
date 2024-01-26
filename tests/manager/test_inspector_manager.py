import pandas as pd
import pytest

from sdgx.data_models.inspectors.manager import InspectorManager
from sdgx.exceptions import InspectorInitError


@pytest.fixture
def manager():
    yield InspectorManager()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.mark.parametrize(
    "basic_inspector",
    ["DiscreteInspector", "NumericInspector", "IDInspector", "BoolInspector", "DatetimeInspector"],
)
def test_manager(basic_inspector, manager: InspectorManager):
    assert manager._normalize_name(basic_inspector) in manager.registed_inspectors


@pytest.mark.parametrize("inspector_name", list(InspectorManager().registed_inspectors.keys()))
def test_inspector(inspector_name, manager: InspectorManager, raw_data: pd.DataFrame):
    each_inspector = manager.init(inspector_name)
    # check type
    assert "sdgx.data_models.inspectors" in str(type(each_inspector))
    # check ready
    assert each_inspector.ready is False
    if not "relationship" in inspector_name:
        each_inspector.fit(raw_data)
        assert each_inspector.ready is True
    # check inspect level property
    assert each_inspector.inspect_level <= 100 and each_inspector.inspect_level > 0
    # check inspect_level.setter
    # set level
    each_inspector.inspect_level = 66
    assert each_inspector.inspect_level == 66
    each_inspector.inspect_level = 88
    assert each_inspector.inspect_level == 88
    each_inspector.inspect_level = 100
    assert each_inspector.inspect_level == 100
    each_inspector.inspect_level = 10
    assert each_inspector.inspect_level == 10
    # test error 1
    has_error = False
    try:
        each_inspector.inspect_level = 101
    except Exception as e:
        has_error = True
        assert type(e) == InspectorInitError
    assert has_error is True
    # test error 2
    has_error = False
    try:
        each_inspector.inspect_level = 0
    except Exception as e:
        has_error = True
        assert type(e) == InspectorInitError
    assert has_error is True


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
