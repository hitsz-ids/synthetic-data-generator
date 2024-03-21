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
from sdgx.data_models.combiner import MetadataCombiner
from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship
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
def demo_relational_table_path(tmp_path):
    dummy_size = 10
    role_set = ["admin", "user", "guest"]

    df = pd.DataFrame(
        {
            "id": list(range(dummy_size)),
            "role": [random.choice(role_set) for _ in range(dummy_size)],
            "name": [ramdon_str() for _ in range(dummy_size)],
            "feature_x": [random.random() for _ in range(dummy_size)],
            "feature_y": [random.random() for _ in range(dummy_size)],
            "feature_z": [random.random() for _ in range(dummy_size)],
        }
    )
    save_path_a = tmp_path / "dummy_relation_A.csv"
    df.to_csv(save_path_a, index=False, header=True)

    sub_size = 5
    assert dummy_size >= sub_size
    df = pd.DataFrame(
        {
            "foreign_id": list(range(sub_size)),
            "feature_i": [random.random() for _ in range(sub_size)],
            "feature_j": [random.random() for _ in range(sub_size)],
            "feature_k": [random.random() for _ in range(sub_size)],
        }
    )
    save_path_b = tmp_path / "dummy_relation_B.csv"
    df.to_csv(save_path_b, index=False, header=True)

    return save_path_a, save_path_b, [("id", "foreign_id")]


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
def demo_multi_table_data_loader(demo_multi_table_data_connector, cacher_kwargs):
    loader_dict = {}
    for each_table in demo_multi_table_data_connector.keys():
        each_connector = demo_multi_table_data_connector[each_table]
        each_d = DataLoader(each_connector, cacher_kwargs=cacher_kwargs)
        loader_dict[each_table] = each_d
    yield loader_dict
    for each_table in demo_multi_table_data_connector.keys():
        demo_multi_table_data_connector[each_table].finalize()


@pytest.fixture
def demo_multi_data_parent_matadata(demo_multi_table_data_loader):
    yield Metadata.from_dataloader(demo_multi_table_data_loader["store"])


@pytest.fixture
def demo_multi_data_child_matadata(demo_multi_table_data_loader):
    yield Metadata.from_dataloader(demo_multi_table_data_loader["train"])


@pytest.fixture
def demo_multi_data_relationship():
    yield Relationship.build(parent_table="store", child_table="train", foreign_keys=["Store"])


@pytest.fixture
def demo_multi_table_data_metadata_combiner(
        demo_multi_data_parent_matadata: Metadata,
        demo_multi_data_child_matadata: Metadata,
        demo_multi_data_relationship: Relationship,
):
    # 1. get metadata
    metadata_dict = {}
    metadata_dict["store"] = demo_multi_data_parent_matadata
    metadata_dict["train"] = demo_multi_data_child_matadata
    # 2. define relationship - already defined
    # 3. define combiner
    m = MetadataCombiner(named_metadata=metadata_dict, relationships=[demo_multi_data_relationship])

    yield m


@pytest.fixture
def get_multi_table_metadata():
    """Return a ``MultiTableMetadata`` object to be used with tests."""
    dict_metadata = {
        'tables': {
            'nesreca': {
                'primary_key': 'id_nesreca',
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'nesreca_val': {'sdtype': 'numerical'}
                }
            },
            'oseba': {
                'columns': {
                    'upravna_enota': {'sdtype': 'id'},
                    'id_nesreca': {'sdtype': 'id'},
                    'oseba_val': {'sdtype': 'numerical'}
                }
            },
            'upravna_enota': {
                'primary_key': 'id_upravna_enota',
                'columns': {
                    'id_upravna_enota': {'sdtype': 'id'},
                    'upravna_val': {'sdtype': 'numerical'}
                }
            }
        },
        'relationships': [
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'nesreca',
                'child_foreign_key': 'upravna_enota'
            },
            {
                'parent_table_name': 'upravna_enota',
                'parent_primary_key': 'id_upravna_enota',
                'child_table_name': 'oseba',
                'child_foreign_key': 'upravna_enota'
            },
            {
                'parent_table_name': 'nesreca',
                'parent_primary_key': 'id_nesreca',
                'child_table_name': 'oseba',
                'child_foreign_key': 'id_nesreca'
            }
        ],
        'METADATA_SPEC_VERSION': 'MULTI_TABLE_V1'
    }

    return MetadataCombiner.load_from_dict(dict_metadata)


@pytest.fixture
def get_multi_table_data():
    """Return a dictionary containing some data for multi table."""
    data = {
        'nesreca': pd.DataFrame({
            'id_nesreca': list(range(4)),
            'upravna_enota': list(range(4)),
            'nesreca_val': list(range(4))
        }),
        'oseba': pd.DataFrame({
            'upravna_enota': list(range(4)),
            'id_nesreca': list(range(4)),
            'oseba_val': list(range(4))
        }),
        'upravna_enota': pd.DataFrame({
            'id_upravna_enota': list(range(4)),
            'upravna_val': list(range(4))
        }),
    }

    return data
