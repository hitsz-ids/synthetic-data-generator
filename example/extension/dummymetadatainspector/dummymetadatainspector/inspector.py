from __future__ import annotations

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class MyOwnInspector(Inspector): ...


@hookimpl
def register(manager):
    manager.register("DummyInspector", MyOwnInspector)
