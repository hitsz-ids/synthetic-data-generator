from __future__ import annotations

import numpy as np 
import pandas as pd 
from typing import Any

from sdgx.data_models.metadata import Metadata  
from sdgx.data_processors.formatters.base import Formatter
from sdgx.data_processors.extension import hookimpl
from sdgx.utils import logger

class DatetimeFormatter(Formatter):
    '''
    Formatter class for handling Datetime formats in pd.DataFrame.
    '''

    datetime_columns: list = None

    datetime_formats: list = None

    default_datetime_format: str = "%Y-%m-%d"

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        '''
        Fit method for datetime formatter, the datetime column and datetime format need to be recorded. 
        
        If there is a column without format, the default format will be used for output (this may cause some problems).
        
        Formatter need to use metadata to record which columns belong to datetime type, and convert timestamp back to datetime type during post-processing.
        '''
        
        # get from metadata 
        self.datetime_columns = metadata.get("datetime_columns")
        self.datetime_formats = metadata.get('datetime_format')
        
        logger.info("DatetimeFormatter Fitted.")
        return 

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Convert method to convert datetime samples into timestamp.

        Args:
            - raw_data (pd.DataFrame): Unprocessed table data
        '''
        if len(self.datetime_columns) == 0:
            logger.info("Converting data using DatetimeFormatter... Finished (No datetime columns).")
            return 
        
        logger.info("Converting data using DatetimeFormatter...")

        res_data = self.convert_datetime_columns(self.datetime_columns, raw_data)

        logger.info("Converting data using DatetimeFormatter... Finished.")

        return res_data

    @staticmethod
    def convert_datetime_columns(datetime_column_list, processed_data):
        """
        Convert datetime columns in processed_data from string to timestamp (int)

        Args:
            - datetime_column_list (list): List of columns that are date time type
            - processed_data (pd.DataFrame): Processed table data

        Returns:
            - result_data (pd.DataFrame): Processed table data with datetime columns converted to timestamp
        """

        # Make a copy of processed_data to avoid modifying the original data
        result_data = processed_data.copy()  

        # Convert each datetime column in datetime_column_list to timestamp
        for column in datetime_column_list:
            # Convert datetime to timestamp (int)
            # TODO may cause error here!
            result_data[column] = pd.to_datetime(result_data[column]).astype(np.int64) # // 10**9

        return result_data
    
    def post_processing(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        '''
        post_processing method for datetime formatter. 
        
        Does not require any action.
        '''
        if len(self.datetime_columns) == 0:
            logger.info("Data reverse-converted by DatetimeFormatter (No datetime columns).")
            return

        result_data = self.convert_timestamp_to_datetime(self.datetime_columns, self.datetime_formats, processed_data, self.default_datetime_format)

        logger.info("Data reverse-converted by DatetimeFormatter.")

        return result_data
    
    @staticmethod
    def convert_timestamp_to_datetime(timestamp_column_list, format_dict, processed_data, default_format):
        """
        Convert timestamp columns to datetime format in a DataFrame.

        Parameters:
            - timestamp_column_list (list): List of column names in the DataFrame which are of timestamp type.
            - datetime_column_dict (dict): Dictionary with column names as keys and datetime format as values.
            - processed_data (pd.DataFrame): DataFrame containing the processed data.

        Returns:
            - result_data (pd.DataFrame): DataFrame with timestamp columns converted to datetime format.
        """

        # Copy the processed data to result_data
        result_data = processed_data.copy()

        # Iterate over each column in the timestamp_column_list
        for column in timestamp_column_list:
            # Check if the column is in the DataFrame
            if column in result_data.columns:
                # Convert the timestamp to datetime format using the format provided in datetime_column_dict
                result_data[column] = pd.to_datetime(result_data[column], unit='s').dt.strftime(format_dict.get(column, default_format))
            else: 
                logger.error(f'Column {column} not in processed data\'s column list!')

        return result_data    

    pass

@hookimpl
def register(manager):
    manager.register("DatetimeFormatter", DatetimeFormatter)

