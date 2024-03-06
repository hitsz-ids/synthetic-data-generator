from __future__ import annotations

from pandas import DataFrame  
from sdgx.data_models.metadata import Metadata  
from sdgx.data_processors.transformers.base import Transformer 
from sdgx.data_processors.extension import hookimpl
from sdgx.utils import logger

class DiscreteTransformer(Transformer):
    '''
    DiscreteTransformer is an important component of sdgx, used to handle discrete columns.
    
    By default, DiscreteTransformer will perform one-hot encoding of discrete columns, and issue a warning message when dimensionality explosion occurs.
    '''

    discrete_columns = None
    '''
    Record which columns are of discrete type.
    '''

    def fit(self, metadata: Metadata | None = None):
        '''
        Fit method for the transformer. 
        '''

        self.discrete_columns = metadata.get('discrete_columns')

        logger.info("DiscreteTransformer Fitted.")
        return 

    def convert(self, raw_data: DataFrame) -> DataFrame:
        '''
        Convert method to handle discrete values in the input data.
        '''

        logger.info("Converting data using DiscreteTransformer...")

        if self.drop:
            res = raw_data.dropna()
        else:
            res = raw_data.fillna(value= self.fill_na_value)  
        
        logger.info("Converting data using DiscreteTransformer... Finished.")

        return res
    
    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        '''
        Reverse_convert method for the transformer. 
        
        
        '''

        logger.info("Data reverse-converted by DiscreteTransformer.")

        return processed_data




    pass