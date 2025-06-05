import random
import string

import numpy as np
import pandas as pd
import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.models.statistics.single_table.tabcop import TabCopSynthesizerModel


def ramdon_str():
    return "".join(random.choice(string.ascii_letters) for _ in range(10))


@pytest.fixture
def single_table_path(tmp_path):
    dummy_size = 10
    role_set = ["admin", "user", "guest"]

    df = pd.DataFrame(
        {
            "role": [random.choice(role_set) for _ in range(dummy_size)],
            "name": [ramdon_str() for _ in range(dummy_size)],
            "feature_x": [random.random() for _ in range(dummy_size)],
            "feature_y": [random.randint(-5, 5) for _ in range(dummy_size)],
            "feature_z": [random.random() for _ in range(dummy_size)],
        }
    )
    df.loc[0:2, "feature_z"] = np.nan
    save_path = tmp_path / "dummy.csv"
    df.to_csv(save_path, index=False, header=True)
    yield save_path
    save_path.unlink()


@pytest.fixture
def single_table_data_connector(single_table_path):
    yield CsvConnector(
        path=single_table_path,
    )


@pytest.fixture
def single_table_data_loader(single_table_data_connector, cacher_kwargs):
    d = DataLoader(single_table_data_connector, cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize()


@pytest.fixture
def single_table_metadata(single_table_data_loader):
    yield Metadata.from_dataloader(single_table_data_loader)


def test_tabcop(single_table_metadata, single_table_data_loader):
    model = TabCopSynthesizerModel(epsilon=1)
    model.fit(single_table_metadata, single_table_data_loader)

    model.save("./model.pkl")
    loaded_model = TabCopSynthesizerModel.load("./model.pkl")

    sampled_data = loaded_model.sample(10)
    original_data = single_table_data_loader.load_all()
    assert len(sampled_data) == 10
    assert sampled_data.columns.tolist() == original_data.columns.tolist()
