import pytest

from sdgx.cachers.manager import CacherManager


@pytest.fixture
def manager():
    yield CacherManager()


def test_registed_data_connector(manager: CacherManager):
    assert manager._normalize_name("DummyCache") in manager.registed_data_connectors
    manager.init_data_connector("DummyCache")


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
