from __future__ import annotations

from typing import Any

from sdgx.cli import exporters
from sdgx.cli.exporters import extension
from sdgx.cli.exporters.base import Exporter
from sdgx.cli.exporters.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class ExporterManager(Manager):
    register_type = Exporter
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_exporters(self):
        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(exporters)

    def init_exporter(self, exporter_name, **kwargs: dict[str, Any]) -> Exporter:
        return self.init(exporter_name, **kwargs)
