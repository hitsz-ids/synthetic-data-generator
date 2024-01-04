import os

os.environ["SDG_NDARRAY_CACHE_ROOT"] = "/tmp/sdgx/ndarray_cache"

import random
import shutil
import string
from functools import partial

import pandas as pd
import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.utils import download_demo_data, download_multi_table_demo_data

_HERE = os.path.dirname(__file__)

# Cache it locally for rerun the tests
DATA_DIR = os.path.join(_HERE, "dataset")


def ramdon_str():
    return "".join(random.choice(string.ascii_letters) for _ in range(10))


@pytest.fixture
def dummy_single_table_path(tmp_path):
    dummy_size = 10
    role_set = ["admin", "user", "guest"]

    df = pd.DataFrame(
        {
            "role": [random.choice(role_set) for _ in range(dummy_size)],
            "name": [ramdon_str() for _ in range(dummy_size)],
            "feature_x": [random.random() for _ in range(dummy_size)],
            "feature_y": [random.random() for _ in range(dummy_size)],
            "feature_z": [random.random() for _ in range(dummy_size)],
        }
    )
    save_path = tmp_path / "dummy.csv"
    df.to_csv(save_path, index=False, header=True)
    yield save_path
    save_path.unlink()


@pytest.fixture
def dummy_single_table_data_connector(dummy_single_table_path):
    yield CsvConnector(
        path=dummy_single_table_path,
    )


@pytest.fixture
def dummy_single_table_data_loader(dummy_single_table_data_connector, cacher_kwargs):
    d = DataLoader(dummy_single_table_data_connector, cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize()


@pytest.fixture
def dummy_single_table_metadata(dummy_single_table_data_loader):
    yield Metadata.from_dataloader(dummy_single_table_data_loader)


@pytest.fixture
def demo_single_table_path():
    yield download_demo_data(DATA_DIR).as_posix()


@pytest.fixture
def cacher_kwargs(tmp_path):
    cache_dir = tmp_path / "cache"
    yield {"cache_dir": cache_dir.as_posix()}
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def dataloader_builder(cacher_kwargs):
    yield partial(DataLoader, cacher_kwargs=cacher_kwargs)


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


@pytest.fixture
def demo_multi_table_path():
    yield download_multi_table_demo_data(DATA_DIR)


@pytest.fixture
def demo_multi_table_data_connector(demo_multi_table_path):
    connector_dict = {}
    for each_table in demo_multi_table_path.keys():
        each_path = demo_multi_table_path[each_table]
        connector_dict[each_table] = CsvConnector(path=each_path)
    yield connector_dict


@pytest.fixture
def demo_single_table_data_loader(demo_multi_table_data_connector, cacher_kwargs):
    loader_dict = {}
    for each_table in demo_multi_table_data_connector.keys():
        each_connector = demo_multi_table_data_connector[each_table]
        each_d = DataLoader(each_connector, cacher_kwargs=cacher_kwargs)
        loader_dict[each_table] = each_d
    yield loader_dict
    for each_table in demo_multi_table_data_connector.keys():
        demo_multi_table_data_connector[each_table].finalize()
