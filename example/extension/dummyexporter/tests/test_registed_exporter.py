import pytest

from sdgx.data_exporters.manager import DataExporterManager


@pytest.fixture
def manager():
    yield DataExporterManager()


def test_registed_exporter(manager: DataExporterManager):
    assert manager._normalize_name("MyOwnExporter") in manager.registed_exporters


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
