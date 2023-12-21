from __future__ import annotations

from typing import Any

from sdgx import data_exporters
from sdgx.data_exporters import extension
from sdgx.data_exporters.base import DataExporter
from sdgx.data_exporters.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class DataExporterManager(Manager):
    register_type = DataExporter
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_exporters(self):
        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(data_exporters)

    def init_exporter(self, exporter_name, **kwargs: dict[str, Any]) -> DataExporter:
        return self.init(exporter_name, **kwargs)
