from __future__ import annotations

from typing import Any

from pandas import DataFrame

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class OutlierTransformer(Transformer):
    """
    
    """

    int_columns: set = set() 
    int_outlier_fill_value = 0

    float_columns: set = set()
    float_outlier_fill_value = 0
    

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        Record int and float columns' name.
        """
        self.int_columns = metadata.int_columns
        self.float_columns = metadata.float_columns
        
        self.fitted = True

        logger.info("OutlierTransformer Fitted.")

    def convert(self, raw_data: DataFrame) -> DataFrame:
        """
        Convert method to handle missing values in the input data.
        """

        res = raw_data

        logger.info("Converting data using OutlierTransformer...")

        # dealing with the int value columns
        def convert_to_int(value):
            try:
                return int(value)
            except ValueError:
                return self.int_outlier_fill_value
            
        for each_col in self.int_columns:
            res[each_col] = res[each_col].apply(convert_to_int)
        
        # then dealing with the float value 
        def convert_to_float(value):
            try:
                return float(value)
            except ValueError:
                return self.float_outlier_fill_value
        
        for each_col in self.float_columns:
            res[each_col] = res[each_col].apply(convert_to_float)
        
        logger.info("Converting data using OutlierTransformer... Finished.")

        return res

    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        """
        Reverse_convert method for the transformer.

        Does not require any action.
        """
        logger.info("Data reverse-converted by OutlierTransformer (No Action).")

        return processed_data


@hookimpl
def register(manager):
    manager.register("OutlierTransformer", OutlierTransformer)
