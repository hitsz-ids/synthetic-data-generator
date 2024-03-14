from __future__ import annotations

import pandas as pd 
from sklearn.preprocessing import OneHotEncoder

from sdgx.data_models.metadata import Metadata  
from sdgx.data_processors.transformers.base import Transformer 
from sdgx.data_processors.extension import hookimpl
from sdgx.utils import logger
from sdgx.data_loader import DataLoader
from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader

class DiscreteTransformer(Transformer):
    '''
    DiscreteTransformer is an important component of sdgx, used to handle discrete columns.
    
    By default, DiscreteTransformer will perform one-hot encoding of discrete columns, and issue a warning message when dimensionality explosion occurs.
    '''

    discrete_columns: list = None
    '''
    Record which columns are of discrete type.
    '''

    encoders: dict = {}

    onehot_encoder_handle_unknown='ignore'

    def fit(self, metadata: Metadata, tabular_data: DataLoader | pd.DataFrame):
        '''
        Fit method for the DiscreteTransformer. 
        '''

        logger.info("Fitting using DiscreteTransformer...")

        self.discrete_columns = metadata.get('discrete_columns')
        
        # no discrete columns 
        if len(self.discrete_columns) == 0 :
            logger.info("Fitting using DiscreteTransformer... Finished (No Columns).")
            return 

        # then, there are >= 1 discrete colums
        for each_col in self.discrete_columns:
            # fit each column 
            self._fit_column(each_col, tabular_data[[each_col]])

        logger.info("Fitting using DiscreteTransformer... Finished.")
        self.fitted = True
        
        return 
    
    def _fit_column(self, column_name: str, column_data: pd.DataFrame):
        '''
        Fit every discrete columns in `_fit_column`.

        Args:
            - column_data (pd.DataFrame): A dataframe containing a column.
            - column_name: str: column name.
        '''

        self.encoders[column_name] = OneHotEncoder(handle_unknown= self.onehot_encoder_handle_unknown)
        # fit the column data
        self.encoders[column_name].fit(column_data)
        
        logger.info(f"Discrete column {column_name} fitted.")
        

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Convert method to handle discrete values in the input data.
        '''

        logger.info("Converting data using DiscreteTransformer...")

        # TODO 
        # transform every discrete column into 
        if len(self.discrete_columns) == 0:
            logger.info("Converting data using DiscreteTransformer... Finished (No column).")
            return 
        
        for each_col in self.discrete_columns:
            new_onehot_column_set = self.encoders[each_col].transform(raw_data[[each_col]])
            # TODO 1- add new_onehot_column_set into the original dataframe
            # TODO 2- delete the original column 
            logger.info(f"Column {each_col} converted.")
        
        logger.info("Converting data using DiscreteTransformer... Finished.")
        
        # return the result
        return 
    
    def _transform_column(self, column_name: str, column_data: pd.DataFrame | pd.Series):
        '''
        Transform every single discrete columns in `_transform_column`.

        Args:
            - column_data (pd.DataFrame): A dataframe containing a column.
            - column_name: str: column name.

        '''
        pass
    
    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Reverse_convert method for the transformer. 
        
        
        '''

        logger.info("Data reverse-converted by DiscreteTransformer.")

        return processed_data

    pass