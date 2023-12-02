from __future__ import annotations

import glob
import importlib
from os.path import basename, dirname, isfile, join
from typing import Any

import pluggy

from sdgx import models
from sdgx.errors import ModelInitializationError, ModelNotFoundError, ModelRegisterError
from sdgx.log import logger
from sdgx.models import extension
from sdgx.models.base import BaseSynthesizerModel
from sdgx.models.extension import project_name as PROJECT_NAME
from sdgx.utils.utils import Singleton


class ModelManager(metaclass=Singleton):
    def __init__(self):
        self.pm = pluggy.PluginManager(PROJECT_NAME)
        self.pm.add_hookspecs(extension)
        self._registed_model: dict[str, type[BaseSynthesizerModel]] = {}

        # Load all
        self.pm.load_setuptools_entrypoints(PROJECT_NAME)

        # Load all local model
        self.load_all_local_model()

    @property
    def registed_model(self):
        # Lazy load when query registed_model
        if self._registed_model:
            return self._registed_model
        for f in self.pm.hook.register(manager=self):
            try:
                f()
            except Exception as e:
                logger.exception(ModelRegisterError(e))
                continue
        return self._registed_model

    def load_all_local_model(self):
        # self.pm.register(sdgx/models/single_table/*)
        self._load_dir(models.single_table)
        # self.pm.register(sdgx/models/multi_tables/*)
        self._load_dir(models.multi_tables)

    def _load_dir(self, module):
        modules = glob.glob(join(dirname(module.__file__), "*.py"))
        sub_packages = (
            basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
        )
        packages = (str(module.__package__) + "." + i for i in sub_packages)
        for p in packages:
            self.pm.register(importlib.import_module(p))

    def _normalize_name(self, model_name: str) -> str:
        return model_name.strip().lower()

    def register(self, model_name, model_cls: type[BaseSynthesizerModel]):
        model_name = self._normalize_name(model_name)
        logger.info(f"Register for new model: {model_name}")
        self._registed_model[model_name] = model_cls

    def init_model(self, model_name, **kwargs: dict[str, Any]) -> BaseSynthesizerModel:
        model_name = self._normalize_name(model_name)
        if not model_name in self.registed_model:
            raise ModelNotFoundError
        try:
            return self.registed_model[model_name](**kwargs)
        except Exception as e:
            raise ModelInitializationError(e)

    @staticmethod
    def load(model_path) -> BaseSynthesizerModel:
        return BaseSynthesizerModel.load(model_path)
