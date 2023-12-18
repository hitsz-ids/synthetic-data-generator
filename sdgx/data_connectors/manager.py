from __future__ import annotations

from typing import Any

from sdgx import data_connectors
from sdgx.data_connectors import extension
from sdgx.data_connectors.base import DataConnector
from sdgx.data_connectors.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class DataConnectorManager(Manager):
    register_type = DataConnector
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_data_connectors(self):
        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(data_connectors)

    def init_data_connector(self, connector_name, **kwargs: dict[str, Any]) -> DataConnector:
        return self.init(connector_name, **kwargs)
