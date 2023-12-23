import pytest

from sdgx.data_exporters.manager import DataExporterManager


@pytest.fixture
def manager():
    yield DataExporterManager()


@pytest.mark.parametrize(
    "supportd_exporter",
    [
        "CsvExporter",
    ],
)
def test_manager(supportd_exporter, manager: DataExporterManager):
    assert manager._normalize_name(supportd_exporter) in manager.registed_exporters


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
