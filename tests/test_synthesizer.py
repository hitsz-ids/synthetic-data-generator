from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.models.base import SynthesizerModel
from sdgx.synthesizer import Synthesizer


class MockModel(SynthesizerModel):
    def fit(self, metadata, dataloader, **kwargs):
        pass

    def sample(self, count, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    def save(self, save_dir: str | Path):
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir.joinpath("mockmoel.pth").touch()

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError
        return MockModel()


class MockDataProcessor(DataProcessor):
    pass


def generator_data() -> pd.DataFrame:
    for i in range(10):
        yield pd.DataFrame({"a": [i], "b": [i * 2]})


class MockDataConnector(GeneratorConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(generator_data, *args, **kwargs)


@pytest.fixture
def metadata():
    yield Metadata()


@pytest.fixture
def synthesizer(cacher_kwargs):
    yield Synthesizer(
        MockModel(),
        data_connector=MockDataConnector(),
        raw_data_loaders_kwargs={"cacher_kwargs": cacher_kwargs},
        data_processors=[MockDataProcessor()],
        processored_data_loaders_kwargs={"cacher_kwargs": cacher_kwargs},
        metadata=Metadata(),
    )


@pytest.fixture
def save_dir(tmp_path):
    d = tmp_path / "unittest-synthesizer"
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_fit(synthesizer):
    synthesizer.fit()


def test_sample(synthesizer):
    assert synthesizer.sample(10) is not None


def test_save_and_load(synthesizer, save_dir):
    assert synthesizer.save(save_dir)
    assert (save_dir / synthesizer.METADATA_SAVE_NAME).exists()
    assert (save_dir / synthesizer.MODEL_SAVE_NAME).exists()

    synthesizer = Synthesizer.load(
        save_dir,
        model=MockModel,
    )
    assert synthesizer


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
