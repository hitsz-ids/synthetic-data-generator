import os

import pytest

from sdgx.utils import download_demo_data

_HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(_HERE, "dataset")


@pytest.fixture
def demo_single_table_path():
    yield download_demo_data(DATA_DIR)


@pytest.fixture
def cacher_kwargs(tmp_path):
    yield {"cache_dir": (tmp_path / "cache").as_posix()}
