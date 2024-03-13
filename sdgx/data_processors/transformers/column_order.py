from __future__ import annotations

import pandas as pd
from typing import Any

from sdgx.data_models.metadata import Metadata  
from sdgx.data_processors.transformers.base import Transformer 
from sdgx.data_processors.extension import hookimpl
from sdgx.utils import logger

class ColumnOrderTransformer(Transformer):
    '''
    Transformer class for handling missing values in data.

    This Transformer is mainly used as a reference for Transformer to facilitate developers to quickly understand the role of Transformer.
    '''

    column_list: list = None
    '''
    The list of tabular data's columns.
    '''

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        '''
        Fit method for the transformer. 
        
        Remember the order of the columns.
        '''

        self.column_list = list(metadata.column_list)

        logger.info("ColumnOrderTransformer Fitted.")

        return 

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Convert method to handle missing values in the input data.
        '''
        logger.info("Converting data using ColumnOrderTransformer...")
        logger.info("Converting data using ColumnOrderTransformer... Finished (No action).")

        return raw_data
    
    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Reverse_convert method for the transformer. 
        '''

        res = self.rearrange_columns(self.column_list, processed_data)
        logger.info("Data reverse-converted by ColumnOrderTransformer.")

        return res
    
    @staticmethod
    def rearrange_columns(column_list, processed_data):
        """
        This method rearranges the columns of a given DataFrame according to the provided column list.
        
        Any columns in the DataFrame that are not in the column list are dropped.

        Args:
            - column_list (list): A list of column names in the order they should appear in the output DataFrame.
            - processed_data (pd.DataFrame): The DataFrame to be rearranged.

        Returns:
            - result_data (pd.DataFrame): The rearranged DataFrame.
        """
        # Use the `reindex` function to rearrange the columns according to `column_list`.
        # The `columns` parameter specifies the order of the columns.
        # The `drop` parameter is set to True to drop any columns not in `column_list`.
        result_data = processed_data.reindex(columns=column_list)
        
        return result_data

    pass

@hookimpl
def register(manager):
    manager.register("ColumnOrderTransformer", ColumnOrderTransformer)

