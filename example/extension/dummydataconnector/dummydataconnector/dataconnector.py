from __future__ import annotations

from sdgx.data_connectors.base import DataConnector


class MyOwnDataConnector(DataConnector): ...


from sdgx.data_connectors.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("DummyDataConnector", MyOwnDataConnector)
