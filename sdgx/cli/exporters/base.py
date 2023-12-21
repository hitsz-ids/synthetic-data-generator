from __future__ import annotations

from typing import Any, Generator

import pandas as pd


class Exporter:
    def write(self, dst: Any, data: pd.DataFrame | Generator[pd.DataFrame, None, None]) -> None:
        raise NotImplementedError
