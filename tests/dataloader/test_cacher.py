from __future__ import annotations

from typing import Generator

import pandas as pd
import pytest

from sdgx.cachers.base import Cacher, NoCache
from sdgx.cachers.disk_cache import DiskCache
from sdgx.data_connectors.csv_connector import CsvConnector


@pytest.fixture
def csv_file(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("index,a,b,c\n0,1,2,3\n1,4,5,6")
    yield csv
    csv.unlink()


class MockCsvConnector(CsvConnector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._readed = False

    @property
    def is_readed(self):
        return self._readed

    def reset(self):
        self._readed = False

    def _read(self, offset=0, limit=0) -> pd.DataFrame:
        self._readed = True
        try:
            return super()._read(offset, limit)
        except Exception:
            self._readed = False
            raise

    def _columns(self) -> list[str]:
        self._readed = True
        try:
            return super()._columns()
        except Exception:
            self._readed = False
            raise

    def iter(self, offset=0, chunksize=0) -> Generator[pd.DataFrame, None, None]:
        self._readed = True
        try:
            return super().iter(offset, chunksize)
        except Exception:
            self._readed = False
            raise


@pytest.fixture
def data_connector(csv_file):
    yield MockCsvConnector(
        path=csv_file,
    )


@pytest.mark.parametrize(
    "cacher_cls",
    [
        NoCache,
        DiskCache,
    ],
)
@pytest.mark.parametrize("blocksize", [1])
@pytest.mark.parametrize("chunksize", [1])
def test_cacher(cacher_cls, cacher_kwargs, blocksize, chunksize, data_connector):
    cacher: Cacher = cacher_cls(blocksize=blocksize, **cacher_kwargs)
    for d in cacher.iter(chunksize, data_connector):
        assert len(d) == chunksize
    if isinstance(cacher, NoCache):
        assert not cacher.is_cached(0)
    else:
        assert cacher.is_cached(0)

        data_connector.reset()
        cacher.load(0, chunksize, data_connector)
        assert not data_connector.is_readed
    assert not cacher.load_all(data_connector).empty


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
