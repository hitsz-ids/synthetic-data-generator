from __future__ import annotations

from sdgx.cli.exporters.base import Exporter


class MyOwnExporter(Exporter):
    ...


from sdgx.cli.exporters.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("MyOwnExporter", MyOwnExporter)
