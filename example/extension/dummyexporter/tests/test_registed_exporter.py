import pytest

from sdgx.cli.exporters.manager import ExporterManager


@pytest.fixture
def manager():
    yield ExporterManager()


def test_registed_exporter(manager: ExporterManager):
    assert manager._normalize_name("MyOwnExporter") in manager.registed_exporters


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
