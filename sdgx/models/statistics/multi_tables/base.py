from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.api import DataFrame as DataFrame

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.log import logger
from sdgx.models.base import SynthesizerModel


# This is the base class of all statistics multi-table models
# When implementing the model, please inherit this class
class MultiTableSynthesizer(SynthesizerModel):

    _parent_map = {}
    _child_map = {}
    _parent_id = {}
    _table_synthesizers = {}
    _foreign_keys = {}

    def fit(self, metadata: Metadata, dataloader: DataLoader, *args, **kwargs):
        """
        Fit the model using the given metadata and dataloader.

        Args:
            metadata (Metadata): The metadata to use.
            dataloader (DataLoader): The dataloader to use.
        """
        raise NotImplementedError


    def sample(self, count: int, *args, **kwargs) -> DataFrame:
        """
        Sample data from the model.

        Args:
            count (int): The number of samples to generate.

        Returns:
            pd.DataFrame: The generated data.
        """

        raise NotImplementedError

    def save(self, save_dir: str | Path):
        pass


    @classmethod
    def load(target_path: str | Path):

        pass


    # first implement these methods
    #              |
    #              V
    def _calculate_parent_map(self):
        pass

    def _calculate_parent_map(self):
        pass

    def _get_foreign_keys(self):

        pass

    def _get_all_foreign_keys(self):

        pass


    def _find_parent_id(self):

        pass


    def get_extended_table(self):
        #

        pass

    def get_extented_rows(self):

        pass


    def _get_num_rows_from_parent(self):

        pass
    #              ^
    #              |
    # first implement these methods

    def check(self):
    ''' Excute necessary checks

    - validate circular relationships
    - validate child map_circular relationship
    - validate all tables connect relationship
    - validate column relationships foreign keys
    '''

        pass



    pass
