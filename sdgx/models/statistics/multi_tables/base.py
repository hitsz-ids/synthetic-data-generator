from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from pydantic import BaseModel
from sdgx.utils import DataAccessType
from sdgx.data_loader import DataLoader
from sdgx.data_models.combiner import MetadataCombiner
from sdgx.log import logger



class MultiTableSynthesizerModel(BaseModel):
    """MultiTableSynthesizerModel
    
    The base model of multi-table statistic models.
    """

    data_access_method: DataAccessType = DataAccessType.pd_data_frame
    '''
    The type of the data access, now support pandas.DataFrame or sdgx.DataLoader.
    '''

    metadata_combiner: MetadataCombiner = None
    """
    metadata_combiner is a sdgx builtin class, it stores all tables' metadata and relationships.
    """

    tables_data_frame: Dict[str, Any] = defaultdict()
    """
    tables_data_frame is a dict contains every table's csv data frame.
    For a small amount of data, this scheme can be used.
    """

    tables_data_loader: Dict[str, Any] = defaultdict()
    """
    tables_data_loader is a dict contains every table's data loader.
    """

    _parent_id: List = []
    """
    _parent_id is used to store all parent table's parimary keys in list.
    """

    _table_synthesizers: Dict[str, Any] = {}
    """
    _table_synthesizers is a dict to store model for each table.
    """

    parent_map: Dict = defaultdict()
    '''
    The mapping from all child tables to their parent table.
    '''

    child_map: Dict = defaultdict()
    '''
    The mapping from all parent tabels to their child table.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._calculate_parent_and_child_map()

        self.check()

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

    def _finalize(self):
        '''Finalize the 
        '''
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
