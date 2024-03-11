from __future__ import annotations

from pandas import DataFrame  
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

    def fit(self, metadata: Metadata | None = None):
        '''
        Fit method for the transformer. 
        
        Remember the order of the columns.
        '''

        self.column_list = list(metadata.column_list)

        logger.info("ColumnOrderTransformer Fitted.")

        return 

    def convert(self, raw_data: DataFrame) -> DataFrame:
        '''
        Convert method to handle missing values in the input data.
        '''
        logger.info("Converting data using ColumnOrderTransformer...")
        logger.info("Converting data using ColumnOrderTransformer... Finished (No action).")

        return raw_data
    
    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        '''
        Reverse_convert method for the transformer. 
        '''


        logger.info("Data reverse-converted by ColumnOrderTransformer.")

        return processed_data

    pass

@hookimpl
def register(manager):
    manager.register("ColumnOrderTransformer", ColumnOrderTransformer)

