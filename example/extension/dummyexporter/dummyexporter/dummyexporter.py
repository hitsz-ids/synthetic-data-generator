from __future__ import annotations

from sdgx.data_exporters.base import DataExporter


class MyOwnExporter(DataExporter): ...


from sdgx.data_exporters.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("MyOwnExporter", MyOwnExporter)
