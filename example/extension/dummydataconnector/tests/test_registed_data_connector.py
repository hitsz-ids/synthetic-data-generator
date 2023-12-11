import pytest

from sdgx.data_connectors.manager import DataConnectorManager


@pytest.fixture
def manager():
    yield DataConnectorManager()


def test_registed_data_connector(manager: DataConnectorManager):
    assert manager._normalize_name("DummyDataConnector") in manager.registed_data_connectors
    manager.init_data_connector("DummyDataConnector")


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
