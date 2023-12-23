from __future__ import annotations

from typing import Any

from sdgx.exceptions import ManagerLoadModelError
from sdgx.manager import Manager
from sdgx.models import extension, ml, statistics
from sdgx.models.base import SynthesizerModel
from sdgx.models.extension import project_name as PROJECT_NAME


class ModelManager(Manager):
    register_type = SynthesizerModel
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_models(self):
        """
        redirect to registed_cls
        """

        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(ml.single_table)
        self._load_dir(ml.multi_tables)
        self._load_dir(statistics.single_table)
        self._load_dir(statistics.multi_tables)

    def init_model(self, model_name, **kwargs: dict[str, Any]) -> SynthesizerModel:
        """
        redirect to init
        """

        return self.init(model_name, **kwargs)

    def load(self, model: type[SynthesizerModel] | str, model_path) -> SynthesizerModel:
        if not (isinstance(model, type) or isinstance(model, str)):
            raise ManagerLoadModelError(
                "model must be type of SynthesizerModel or str for model_name"
            )
        if isinstance(model, str):
            model = self._normalize_name(model)

        if isinstance(model, str) and model not in self.registed_models:
            raise ManagerLoadModelError(f"{model} is not registered.")
        model = model if isinstance(model, type) else self.registed_models[model]

        try:
            return model.load(model_path)
        except Exception as e:
            raise ManagerLoadModelError(e)
