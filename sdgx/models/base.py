from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.exceptions import SynthesizerInitError


class SynthesizerModel:
    use_dataloader: bool = False
    use_raw_data: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # specify data access type
        if "use_dataloader" in kwargs.keys():
            self.use_dataloader = kwargs["use_dataloader"]
        if "use_raw_data" in kwargs.keys():
            self.use_raw_data = kwargs["use_raw_data"]

    def _check_access_type(self):
        if self.use_dataloader == self.use_raw_data == False:
            raise SynthesizerInitError(
                "Data access type not specified, please use `use_raw_data: bool` or `use_dataloader: bool` to specify data access type."
            )
        elif self.use_dataloader == self.use_raw_data == True:
            raise SynthesizerInitError("Duplicate data access type found.")

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
