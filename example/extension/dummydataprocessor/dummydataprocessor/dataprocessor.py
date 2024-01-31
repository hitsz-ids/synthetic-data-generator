from __future__ import annotations

from sdgx.data_processors.base import DataProcessor


class MyOwnDataProcessor(DataProcessor): ...


from sdgx.data_processors.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("DummyDataProcessor", MyOwnDataProcessor)
