import pytest

from sdgx.data_models.inspectors.manager import InspectorManager


@pytest.fixture
def manager():
    yield InspectorManager()


@pytest.mark.parametrize(
    "supported_inspector",
    [],
)
def test_manager(supported_inspector, manager: InspectorManager):
    assert manager._normalize_name(supported_inspector) in manager.registed_inspectors


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
