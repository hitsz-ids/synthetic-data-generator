from __future__ import annotations

from typing import Generator

import pandas as pd


class DataConnector:
    """
    DataConnector warps data source into ``pd.DataFrame``.

    For different data source, implement a specific subclass.
    """

    identity = None
    """
    Identity of data source, e.g. table name, hash of content
    """

    def _read(self, offset: int = 0, limit: int | None = None) -> pd.DataFrame | None | None:
        """
        Subclass must implement this for reading data.

        See ``read`` for more details.
        """
        raise NotImplementedError

    def _columns(self) -> list[str]:
        """
        Subclass should implement this for reading columns if there is an efficient way for peaking columns.

        See ``column`` for more details.
        """
        raise NotImplementedError

    def _iter(self, offset: int = 0, chunksize: int = 0) -> Generator[pd.DataFrame, None, None]:
        """
        Subclass should implement this for reading data in chunk.

        See ``iter`` for more details.
        """
        raise NotImplementedError

    def iter(self, offset: int = 0, chunksize: int = 0) -> Generator[pd.DataFrame, None, None]:
        """
        Interface for reading data in chunk.

        Args:
            offset (int, optional): Offset for reading. Defaults to 0.
            chunksize (int, optional): Chunksize for reading. Defaults to 0.

        Returns:
            typing.Generator[pd.DataFrame, None, None]: Generator/Iterator for readed dataframe
        """
        return self._iter(offset, chunksize)

    def read(self, offset: int = 0, limit: int | None = None) -> pd.DataFrame | None:
        """
        Interface for reading data.

        Args:
            offset (int, optional): Offset for reading. Defaults to 0.
            limit (int, optional): Limit for reading. Defaults to None.
                None is for reading all data and 0 is for reading no data(only header).

        Returns:
            pd.DataFrame: Readed dataframe
        """
        return self._read(offset, limit)

    def columns(self) -> list[str]:
        """
        Interface for peaking columns.
        """
        try:
            return self._columns()
        except NotImplementedError:
            return self.read(0, 1).columns.tolist()

    def keys(self) -> list[str]:
        """
        Same as ``columns``.
        """
        return self.columns()

    def finalize(self):
        """
        Finalize the data connector.
        """
        pass
