from __future__ import annotations

import glob
import importlib
from os.path import basename, dirname, isfile, join
from typing import Any

import pluggy

from sdgx import models
from sdgx.exceptions import InitializationError, NotFoundError, RegisterError
from sdgx.utils import Singleton, logger


class Manager(metaclass=Singleton):
    """
    Base class for all manager.

    Manager is a singleton class for preventing multiple initialization.

    Define following attributes in subclass:
        * register_type: Base class for registered class
        * project_name: Name of entry-point for extensio
        * hookspecs_model: Hook specification model(where @hookspec is defined)

    For available managers, please refer to :ref:`Plugin-supported modules`

    """

    register_type: type = object
    """
    Base class for registered class
    """
    project_name: str = ""
    """
    Name of entry-point for extension
    """

    hookspecs_model = None
    """
    Hook specification model(where @hookspec is defined)
    """

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
    def registed_cls(self) -> dict[str, type]:
        """
        Access all registed class.

        Lazy load, only load once.
        """
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
        """
        Import all python files in a submodule.
        """
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
        """
        Register a new model, if the model is already registed, skip it.
        """

        cls_name = self._normalize_name(cls_name)
        logger.debug(f"Register for new model: {cls_name}")
        if cls in self._registed_cls.values():
            logger.error(f"SKIP: {cls_name} is already registed")
            return
        if not issubclass(cls, self.register_type):
            logger.error(f"SKIP: {cls_name} is not a subclass of {self.register_type}")
            return
        self._registed_cls[cls_name] = cls

    def init(self, c, **kwargs: dict[str, Any]):
        """
        Init a new subclass of self.register_type.

        Raises:
            NotFoundError: if cls_name is not registered
            InitializationError: if failed to initialize
        """
        if isinstance(c, self.register_type):
            return c

        if isinstance(c, type):
            cls_type = c
        else:
            c = self._normalize_name(c)

            if not c in self.registed_cls:
                raise NotFoundError
            cls_type = self.registed_cls[c]
        try:
            instance = cls_type(**kwargs)
            if not isinstance(instance, self.register_type):
                raise InitializationError(f"{c} is not a subclass of {self.register_type}.")
            return instance
        except Exception as e:
            raise InitializationError(e)
