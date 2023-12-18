import pytest

from sdgx.data_connectors.manager import DataConnectorManager


@pytest.fixture
def manager():
    yield DataConnectorManager()


def test_registed_data_connector(manager: DataConnectorManager):
    assert manager._normalize_name("DummyDataConnector") in manager.registed_data_connectors


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
