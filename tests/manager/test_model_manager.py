import pytest

from sdgx.models.manager import ModelManager


@pytest.fixture
def manager():
    yield ModelManager()


@pytest.mark.parametrize(
    "supported_model",
    [
        "ctgan",
    ],
)
def test_manager(supported_model, manager: ModelManager):
    assert manager._normalize_name(supported_model) in manager.registed_models
    manager.init_model(supported_model, epochs=1)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
