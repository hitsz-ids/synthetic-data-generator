import pytest

from sdgx.models.manager import ModelManager


@pytest.fixture
def manager():
    yield ModelManager()


def test_registed_model(manager: ModelManager):
    assert manager._normalize_name("DummyModel") in manager.registed_models


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
