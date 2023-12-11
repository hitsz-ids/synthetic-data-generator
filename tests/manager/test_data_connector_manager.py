import pytest

from sdgx.data_connectors.manager import DataConnectorManager


@pytest.fixture
def manager():
    yield DataConnectorManager()


@pytest.mark.parametrize(
    "supported_data_connector",
    [],
)
def test_manager(supported_data_connector, manager: DataConnectorManager):
    assert manager._normalize_name(supported_data_connector) in manager.registed_data_connectors
    manager.init_data_connector(supported_data_connector)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
