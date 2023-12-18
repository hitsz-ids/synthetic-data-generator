import pytest

from sdgx.cachers.manager import CacherManager


@pytest.fixture
def manager():
    yield CacherManager()


def test_registed_cacher(manager: CacherManager):
    assert manager._normalize_name("DummyCache") in manager.registed_cachers


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
