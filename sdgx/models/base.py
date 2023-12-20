from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata


class SynthesizerModel:
    def fit(self, metadata: Metadata, dataloader: DataLoader, *args, **kwargs):
        """
        Fit the model using the given metadata and dataloader.

        Args:
            metadata (Metadata): The metadata to use.
            dataloader (DataLoader): The dataloader to use.
        """
        raise NotImplementedError

    def sample(self, count: int, *args, **kwargs) -> pd.DataFrame:
        """
        Sample data from the model.

        Args:
            count (int): The number of samples to generate.

        Returns:
            pd.DataFrame: The generated data.
        """

        raise NotImplementedError

    def save(self, save_dir: str | Path):
        """
        Dump model to file.

        Args:
            save_dir (str | Path): The directory to save the model.
        """

        raise NotImplementedError

    @classmethod
    def load(cls, save_dir: str | Path) -> "SynthesizerModel":
        """
        Load model from file.

        Args:
            save_dir (str | Path): The directory to load the model from.
        """
        raise NotImplementedError
