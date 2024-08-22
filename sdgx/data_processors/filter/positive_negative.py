from __future__ import annotations
from typing import Any
import pandas as pd

from sdgx.data_processors.filter.base import Filter
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.utils import logger

class PositiveNegativeFilter(Filter):
    """
    Add docstring here.

    Attributes:
        # add attribute here. 
    """

    int_columns: set = set()
    """
    A set of column names that contain integer values.
    """

    float_columns: set = set()
    """
    A set of column names that contain float values.
    """

    positive_columns: set = set() 
    '''
    A set of column names that are identified as containing positive numeric values.
    '''

    negative_columns: set = set() 
    '''
    A set of column names that are identified as containing negative numeric values.
    '''


    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the data filter.
        """
        logger.info("PositiveNegativeFilter Fitted.")

        # record int and float data 
        self.int_columns = metadata.int_columns
        self.float_columns = metadata.float_columns 

        # record pos and neg 
        self.positive_columns = set(metadata.numeric_format['positive'])
        self.negative_columns = set(metadata.numeric_format['negative'])
    

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert method for data filter (No Action). 
        """

        logger.info("Converting data using PositiveNegativeFilter... Finished (No Action)")

        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse_convert method for the pos_neg data filter.
        """

        logger.info(f"Data reverse-converted by PositiveNegativeFilter Start with Shape: {processed_data.shape}.")

        

        logger.info(f"Data reverse-converted by PositiveNegativeFilter with Output Shape: {processed_data.shape}.")

        return processed_data

@hookimpl
def register(manager):
    manager.register("PositiveNegativeFilter", PositiveNegativeFilter)
