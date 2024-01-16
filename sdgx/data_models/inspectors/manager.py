from __future__ import annotations

from typing import Any, Iterable

from sdgx.data_models import inspectors
from sdgx.data_models.inspectors import extension
from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import project_name as PROJECT_NAME
from sdgx.manager import Manager


class InspectorManager(Manager):
    register_type = Inspector
    project_name = PROJECT_NAME
    hookspecs_model = extension

    @property
    def registed_inspectors(self):
        return self.registed_cls

    def load_all_local_model(self):
        self._load_dir(inspectors)

    def init_all_inspectors(self, **kwargs: Any) -> list[Inspector]:
        return [
            self.init(inspector_name, **kwargs)
            for inspector_name in self.registed_inspectors.keys()
        ]

    def init_inspcetors(
        self,
        includes: Iterable[str] | None = None,
        excludes: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> list[Inspector]:
        includes = includes or self.registed_inspectors.keys()
        if excludes:
            includes = list(set(includes) - set(excludes))
        return [self.init(inspector_name, **kwargs) for inspector_name in includes]
