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


def test_gaussian_copula(dummy_data, discrete_cols):
    model = GaussianCopulaSynthesizer(discrete_cols)
    model.fit(dummy_data)

    sampled_data = model.sample(10)
    assert len(sampled_data) == 10
    assert sampled_data.columns.tolist() == dummy_data.columns.tolist()
