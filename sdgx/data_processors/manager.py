from __future__ import annotations

from typing import Any

from sdgx import data_processors
from sdgx.data_processors import extension
from sdgx.data_processors.base import DataProcessor
from sdgx.data_processors.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class DataProcessorManager(Manager):
    register_type = DataProcessor
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_data_processors(self):
        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(data_processors.formatters)
        self._load_dir(data_processors.generators)
        self._load_dir(data_processors.samplers)
        self._load_dir(data_processors.transformers)

    def init_data_processor(self, processor_name, **kwargs: dict[str, Any]) -> DataProcessor:
        return self.init(processor_name, **kwargs)
