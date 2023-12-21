import os

os.environ["SDG_NDARRAY_CACHE_ROOT"] = "/tmp/sdgx/ndarray_cache"
import shutil

import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata.base import Metadata
from sdgx.utils import download_demo_data

_HERE = os.path.dirname(__file__)

# Cache it locally for rerun the tests
DATA_DIR = os.path.join(_HERE, "dataset")


@pytest.fixture
def demo_single_table_path():
    yield download_demo_data(DATA_DIR)


@pytest.fixture
def cacher_kwargs(tmp_path):
    cache_dir = tmp_path / "cache"
    yield {"cache_dir": cache_dir.as_posix()}
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def demo_single_table_data_connector(demo_single_table_path):
    yield CsvConnector(
        path=demo_single_table_path,
    )


@pytest.fixture
def demo_single_table_data_loader(demo_single_table_data_connector, cacher_kwargs):
    d = DataLoader(demo_single_table_data_connector, cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize()


@pytest.fixture
def demo_single_table_metadata(demo_single_table_data_loader):
    yield Metadata.from_dataloader(demo_single_table_data_loader)
