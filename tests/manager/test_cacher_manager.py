import pytest

from sdgx.cachers.manager import CacherManager


@pytest.fixture
def manager():
    yield CacherManager()


@pytest.mark.parametrize(
    "supportd_cacher",
    ["NoCache", "DiskCache"],
)
def test_manager(supportd_cacher, manager: CacherManager):
    assert manager._normalize_name(supportd_cacher) in manager.registed_cachers


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
