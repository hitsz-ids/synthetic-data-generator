from __future__ import annotations

from typing import Any

from sdgx import cachers
from sdgx.cachers import extension
from sdgx.cachers.base import Cacher
from sdgx.cachers.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class CacherManager(Manager):
    register_type = Cacher
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_cachers(self):
        """
        redirect to registed_cls
        """
        return self.registed_cls

    def load_all_local_model(self):
        """
        Load all local model. Currently only ``sdgx.cachers``.
        """

        self._load_dir(cachers)

    def init_cacher(self, cacher_name, **kwargs: dict[str, Any]) -> Cacher:
        """
        redirect to init
        """
        return self.init(cacher_name, **kwargs)
