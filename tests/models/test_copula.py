from pathlib import Path

import pandas as pd
import pytest

from sdgx.models.statistics.single_table.copula import GaussianCopulaSynthesizer
from sdgx.utils import get_demo_single_table


@pytest.fixture
def dummy_data(dummy_single_table_path):
    yield pd.read_csv(dummy_single_table_path)


@pytest.fixture
def discrete_cols(dummy_data):
    yield [col for col in dummy_data.columns if not col.startswith("feature")]


def test_gaussian_copula(dummy_single_table_metadata, dummy_single_table_data_loader):
    model = GaussianCopulaSynthesizer()
    model.discrete_cols = discrete_cols
    model.fit(dummy_single_table_metadata, dummy_single_table_data_loader)

    sampled_data = model.sample(10)
    original_data = dummy_single_table_data_loader.load_all()
    assert len(sampled_data) == 10
    assert sampled_data.columns.tolist() == original_data.columns.tolist()
