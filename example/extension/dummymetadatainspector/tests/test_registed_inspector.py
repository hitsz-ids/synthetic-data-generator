import pytest

from sdgx.data_models.inspectors.manager import InspectorManager


@pytest.fixture
def manager():
    yield InspectorManager()


def test_registed_cacher(manager: InspectorManager):
    assert manager._normalize_name("DummyInspector") in manager.registed_inspectors


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
