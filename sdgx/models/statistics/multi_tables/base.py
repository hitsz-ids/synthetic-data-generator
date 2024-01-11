from __future__ import annotations

import numpy as np
import pandas as pd

import time
from pathlib import Path
from typing import Any, Dict, Set


from pandas.core.api import DataFrame as DataFrame

from pydantic import BaseModel
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.log import logger

from sdgx.data_models.combiner import MetadataCombiner
from sdgx.data_models.relationship import Relationship

class MultiTableSynthesizerModel(BaseModel): 

    
    
    _parent_id = {}
    _table_synthesizers = {}
    _foreign_keys = {}
    
    metadata_combiner: MetadataCombiner = None
    parent_map: Dict = {} 
    child_map: Dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._calculate_parent_map()

        # self.check()
        pass

    # first implement these methods
    #              |
    #              V
    def _calculate_parent_and_child_map(self):
        '''Get the mapping from all parent tables to self._parent_map
        - key(str) is a child map;
        - value(str) is the parent map.
        '''
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            parent_table = each_relationship.parent_table
            child_table  = each_relationship.child_table
            self.parent_map[child_table] = parent_table
            self.child_map [parent_table] = child_table

    def _get_foreign_keys(self, parent_table, child_table):
        '''Get the foreign key list from a relationship 
        '''
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            # find the exact relationship and return foreign keys 
            if each_relationship.parent_table == parent_table and \
               each_relationship.child_table  == child_table:
                return  each_relationship.foreign_keys
        pass

    def _get_all_foreign_keys(self, child_table): 
        '''Given a child table, return ALL foreign keys from metadata.
        '''
        all_foreign_keys = [] 
        relationships = self.metadata_combiner.relationships
        for each_relationship in relationships:
            # find the exact relationship and return foreign keys 
            if each_relationship.child_table  == child_table:
                all_foreign_keys.append(each_relationship.foreign_keys)

        return all_foreign_keys

    # (?)
    # seems cannot implement this in base class 
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

    def check(self, check_circular = True): 
        ''' Excute necessary checks
        
        - validate circular relationships
        - validate child map_circular relationship
        - validate all tables connect relationship
        - validate column relationships foreign keys
        '''
        
        pass 

    def fit(self, dataloader: DataLoader, *args, **kwargs):
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
    pass