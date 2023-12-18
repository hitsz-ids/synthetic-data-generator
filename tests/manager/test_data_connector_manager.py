import pytest

from sdgx.data_connectors.manager import DataConnectorManager


@pytest.fixture
def manager():
    yield DataConnectorManager()


@pytest.mark.parametrize(
    "supported_data_connector",
    ["CsvConnector"],
)
def test_manager(supported_data_connector, manager: DataConnectorManager):
    assert manager._normalize_name(supported_data_connector) in manager.registed_data_connectors


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
