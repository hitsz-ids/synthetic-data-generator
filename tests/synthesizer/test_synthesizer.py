from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.models.base import SynthesizerModel
from sdgx.synthesizer import Synthesizer


class MockModel(SynthesizerModel):
    MODEL_SAVE_NAME = "mockmoel.pth"

    def fit(self, metadata, dataloader, **kwargs):
        pass

    def sample(self, count, **kwargs) -> pd.DataFrame:
        return pd.DataFrame({"a": [i for i in range(count)], "b": [i * 2 for i in range(count)]})

    def save(self, save_dir: str | Path):
        save_dir = Path(save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir.joinpath(self.MODEL_SAVE_NAME).touch()

    @classmethod
    def load(cls, save_dir: str | Path):
        save_dir = Path(save_dir).expanduser().resolve()
        if not save_dir.joinpath(cls.MODEL_SAVE_NAME).exists():
            raise FileNotFoundError
        return MockModel()


class MockDataProcessor(DataProcessor):
    fitted = True

    def _fit(self, metadata: Metadata | None = None, **kwargs: Dict[str, Any]):
        return

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
        processed_data_loaders_kwargs={"cacher_kwargs": cacher_kwargs},
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
    assert len(synthesizer.sample(10)) == 10
    for df in synthesizer.sample(10, chunksize=5):
        assert len(df) == 5


def test_save_and_load(synthesizer, save_dir):
    assert synthesizer.save(save_dir)
    assert (save_dir / synthesizer.METADATA_SAVE_NAME).exists()
    assert (save_dir / synthesizer.MODEL_SAVE_DIR).exists()

    synthesizer = Synthesizer.load(
        save_dir,
        model=MockModel,
    )
    assert synthesizer


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
