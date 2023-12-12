from __future__ import annotations

from typing import Generator

import pandas as pd


class DataConnector:
    def _read(self, offset=0, limit=0) -> pd.DataFrame:
        raise NotImplementedError

    def _columns(self) -> list[str]:
        raise NotImplementedError

    def iter(self, offset=0, chunksize=0) -> Generator[pd.DataFrame, None, None]:
        raise NotImplementedError

    def read(self, offset=0, limit=None) -> pd.DataFrame:
        return self._read(offset, limit)

    def columns(self) -> list[str]:
        try:
            return self._columns()
        except NotImplementedError:
            return self.read(0, 1).columns.tolist()

    def keys(self) -> list[str]:
        return self.columns()
