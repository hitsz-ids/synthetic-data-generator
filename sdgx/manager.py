from __future__ import annotations

import glob
import importlib
from os.path import basename, dirname, isfile, join
from typing import Any

import pluggy

from sdgx import models
from sdgx.exceptions import InitializationError, NotFoundError, RegisterError
from sdgx.log import logger
from sdgx.utils.utils import Singleton


class Manager(metaclass=Singleton):
    register_type = object
    project_name = ""
    hookspecs_model = None

    def __init__(self):
        self.pm = pluggy.PluginManager(self.project_name)
        self.pm.add_hookspecs(self.hookspecs_model)
        self._registed_cls: dict[str, type[self.register_type]] = {}

        # Load all
        self.pm.load_setuptools_entrypoints(self.project_name)

        # Load all local model
        self.load_all_local_model()

    def load_all_local_model(self):
        """
        Implement this function to load all local model
        """
        return

    @property
    def registed_cls(self):
        # Lazy load when query registed_models
        if self._registed_cls:
            return self._registed_cls
        for f in self.pm.hook.register(manager=self):
            try:
                f()
            except Exception as e:
                logger.exception(RegisterError(e))
                continue
        return self._registed_cls

    def _load_dir(self, module):
        modules = glob.glob(join(dirname(module.__file__), "*.py"))
        sub_packages = (
            basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
        )
        packages = (str(module.__package__) + "." + i for i in sub_packages)
        for p in packages:
            self.pm.register(importlib.import_module(p))

    def _normalize_name(self, name: str) -> str:
        return name.strip().lower()

    def register(self, cls_name, cls: type):
        cls_name = self._normalize_name(cls_name)
        logger.info(f"Register for new model: {cls_name}")
        if cls in self._registed_cls.values():
            logger.error(f"SKIP: {cls_name} is already registed")
            return
        if not issubclass(cls, self.register_type):
            logger.error(f"SKIP: {cls_name} is not a subclass of {self.register_type}")
            return
        self._registed_cls[cls_name] = cls

    def init(self, cls_name, **kwargs: dict[str, Any]):
        cls_name = self._normalize_name(cls_name)
        if not cls_name in self.registed_models:
            raise NotFoundError
        try:
            instance = self.registed_cls[cls_name](**kwargs)
            if not isinstance(instance, self.register_type):
                raise InitializationError(
                    f"{cls_name} is not a subclass of {self.registed_models}."
                )
            return instance
        except Exception as e:
            raise InitializationError(e)
