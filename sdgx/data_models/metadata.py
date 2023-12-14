from __future__ import annotations

from enum import Enum
from typing import List

import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader


class DType(Enum):
    datetime = "datetime"
    timestamp = "timestamp"
    numeric = "numeric"
    category = "category"


class Relationship:
    pass


class Metadata(BaseModel):
    # fields: List[str]

    @classmethod
    def from_dataloader(dataloader: DataLoader) -> "Metadata":
        return Metadata()

    @classmethod
    def from_dataframe(df: pd.DataFrame) -> "Metadata":
        return Metadata()
