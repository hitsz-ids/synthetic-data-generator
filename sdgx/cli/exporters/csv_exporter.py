from sdgx.cli.exporters.base import Exporter


class CsvExporter(Exporter):
    ...


from sdgx.cli.exporters.extension import hookimpl


@hookimpl
def register(manager):
    manager.register("CsvExporter", CsvExporter)
