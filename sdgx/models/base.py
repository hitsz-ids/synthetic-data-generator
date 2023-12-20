from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata


class SynthesizerModel:
    def fit(self, metadata: Metadata, dataloader: DataLoader, *args, **kwargs):
        raise NotImplementedError

    def sample(self, count: int, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def save(self, save_dir: str | Path):
        raise NotImplementedError

    @classmethod
    def load(cls, save_dir: str | Path) -> "SynthesizerModel":
        raise NotImplementedError
