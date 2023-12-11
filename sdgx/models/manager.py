from __future__ import annotations

from typing import Any

from sdgx import models
from sdgx.manager import Manager
from sdgx.models import extension
from sdgx.models.base import SynthesizerModel
from sdgx.models.extension import project_name as PROJECT_NAME


class ModelManager(Manager):
    register_type = SynthesizerModel
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_models(self):
        """
        Proxy to registed_cls
        """

        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(models.single_table)
        self._load_dir(models.multi_tables)

    def init_model(self, model_name, **kwargs: dict[str, Any]) -> SynthesizerModel:
        """
        Proxy to init
        """

        return self.init(model_name, **kwargs)

    @staticmethod
    def load(model_path) -> SynthesizerModel:
        return SynthesizerModel.load(model_path)
