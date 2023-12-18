import pytest

from sdgx.data_processors.manager import DataProcessorManager


@pytest.fixture
def manager():
    yield DataProcessorManager()


def test_registed_data_processor(manager: DataProcessorManager):
    assert manager._normalize_name("DummyDataProcessor") in manager.registed_data_processors


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
