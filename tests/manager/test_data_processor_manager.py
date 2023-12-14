import pytest

from sdgx.data_processors.manager import DataProcessorManager


@pytest.fixture
def manager():
    yield DataProcessorManager()


@pytest.mark.parametrize(
    "supported_data_processor",
    [],
)
def test_manager(supported_data_processor, manager: DataProcessorManager):
    assert manager._normalize_name(supported_data_processor) in manager.registed_data_processors


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
