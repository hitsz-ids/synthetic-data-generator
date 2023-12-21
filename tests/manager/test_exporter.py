import pytest

from sdgx.cli.exporters.manager import ExporterManager


@pytest.fixture
def manager():
    yield ExporterManager()


@pytest.mark.parametrize(
    "supportd_exporter",
    [
        "CsvExporter",
    ],
)
def test_manager(supportd_exporter, manager: ExporterManager):
    assert manager._normalize_name(supportd_exporter) in manager.registed_exporters


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
