from __future__ import annotations

from typing import Generator

import pandas as pd


class Exporter:
    def write(self, data: pd.DataFrame | Generator[pd.DataFrame, None, None]) -> None:
        raise NotImplementedError
