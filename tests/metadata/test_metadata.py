import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata.base import Metadata


@pytest.fixture
def dataloader(demo_single_table_path, cacher_kwargs):
    d = DataLoader(CsvConnector(path=demo_single_table_path), cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize(clear_cache=True)


@pytest.fixture
def metadata(dataloader):
    yield Metadata.from_dataloader(dataloader)


def test_metadata(metadata: Metadata):
    assert metadata.discrete_columns == metadata.get("discrete_columns")
    assert metadata.model_dump_json()


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
