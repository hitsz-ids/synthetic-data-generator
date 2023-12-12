from sdgx.data_connectors.base import DataConnector


class DataLoader:
    """
    Wrapper of :ref:`DataConnector`
    """

    def __init__(self, data_connector: DataConnector) -> None:
        self.data_connector = data_connector
