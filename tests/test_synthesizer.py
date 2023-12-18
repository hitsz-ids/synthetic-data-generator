from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sdgx.data_connectors.generator_connector import GeneratorConnector
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

    def load(self, path: str | Path):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError
        return


def generator_data() -> pd.DataFrame:
    for i in range(10):
        yield pd.DataFrame({"a": [i], "b": [i * 2]})


class MockDataConnector(GeneratorConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(generator_data, *args, **kwargs)


@pytest.fixture
def synthesizer(cacher_kwargs):
    yield Synthesizer(
        MockModel(),
        data_connector=MockDataConnector(),
        raw_data_loaders_kwargs={"cacher_kwargs": cacher_kwargs},
        processored_data_loaders_kwargs={"cacher_kwargs": cacher_kwargs},
    )


def test_fit(synthesizer):
    synthesizer.fit()


def test_sample(synthesizer):
    assert synthesizer.sample(10)


@pytest.mark.xfail
def test_save(synthesizer):
    assert synthesizer.save()


@pytest.mark.xfail
def test_load():
    synthesizer = Synthesizer.load()
    assert synthesizer


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
