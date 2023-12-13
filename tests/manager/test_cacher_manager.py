import pytest

from sdgx.cachers.manager import CacherManager


@pytest.fixture
def manager():
    yield CacherManager()


@pytest.mark.parametrize(
    "supported_data_connector",
    ["NoCache", "MemoryCache", "DiskCache"],
)
def test_manager(supported_data_connector, manager: CacherManager):
    assert manager._normalize_name(supported_data_connector) in manager.registed_cacher


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
