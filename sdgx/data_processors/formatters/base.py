from __future__ import annotations

import pandas as pd 
from sdgx.data_processors.base import DataProcessor


class Formatter(DataProcessor):
    """
    Base class for formatters.

    Formatter is used to convert data from one format to another.

    For example, parse datetime into timestamp when trainning,
    and format timestamp into datetime when sampling.

    Difference with :ref:`Transformer`:
    - When a single column is used as input, use formatter for formatting issues.
    - When a whole table is used as input, use :ref:`Transformer`.
    - :ref:`Transformer` sometimes implements some functions with the help of Formatter.

    """

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Convert processed data into raw data.

        Args:
            processed_data (pd.DataFrame): Processed data

        Returns:
            pd.DataFrame: Raw data
        """
        return self.post_processing(processed_data)
    
     
    def post_processing(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        '''
        For formatter, please rewrite this method.
        
        Args:
            processed_data (pd.DataFrame): Processed data

        Returns:
            pd.DataFrame: Raw data
        '''

        return processed_data
