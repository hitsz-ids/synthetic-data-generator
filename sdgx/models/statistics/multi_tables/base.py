from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sdgx.data_loader import DataLoader
from sdgx.data_models.combiner import MetadataCombiner
from sdgx.data_models.metadata import Metadata
from sdgx.data_models.relationship import Relationship
from sdgx.log import logger


class MultiTableSynthesizerModel(BaseModel):
    _parent_id = {}
    _table_synthesizers = {}
    _foreign_keys = {}

    metadata_combiner: MetadataCombiner = None
    parent_map: Dict = {}
    child_map: Dict = {}
    _augmented_tables: List = []

    tables_data_loader: Dict(str, DataLoader)
    '''
    tables_data_loader is a dict contains every table's data loader.
    '''
    


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._calculate_parent_and_child_map()

        self.check()

    
    # first implement these methods
    #              |
    #              V
    def _calculate_parent_and_child_map(self):
        """Get the mapping from all parent tables to self._parent_map
        - key(str) is a child map;
        - value(str) is the parent map.
        """
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            parent_table = each_relationship.parent_table
            child_table = each_relationship.child_table
            self.parent_map[child_table] = parent_table
            self.child_map[parent_table] = child_table

    def _get_foreign_keys(self, parent_table, child_table):
        """Get the foreign key list from a relationship"""
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            # find the exact relationship and return foreign keys
            if (
                each_relationship.parent_table == parent_table
                and each_relationship.child_table == child_table
            ):
                return each_relationship.foreign_keys
        return []

    def _get_all_foreign_keys(self, child_table):
        """Given a child table, return ALL foreign keys from metadata."""
        all_foreign_keys = []
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            # find the exact relationship and return foreign keys
            if each_relationship.child_table == child_table:
                all_foreign_keys.append(each_relationship.foreign_keys)

        return all_foreign_keys
    

    def _get_num_rows_from_parent(self):
        pass

    #              ^
    #              |
    # first we implement these methods

    def calculate_table_likehoods(
        self, child_table_rows, parent_table_rows, child_table_name, foreign_key
    ):
        raise NotImplementedError

    def check(self, check_circular=True):
        """Excute necessary checks

        - validate circular relationships
        - validate child map_circular relationship
        - validate all tables connect relationship
        - validate column relationships foreign keys
        """

        pass

    def fit(self, dataloader: DataLoader, *args, **kwargs):
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
        pass

    @classmethod
    def load(target_path: str | Path):
        pass

    pass
