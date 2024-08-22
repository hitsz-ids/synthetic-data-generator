from __future__ import annotations
from typing import Any
import pandas as pd

from sdgx.data_processors.filter.base import Filter
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.utils import logger

class PositiveNegativeFilter(Filter):
    """
    一个用于过滤正负值的数据处理器。

    此过滤器用于确保特定列中的值保持正值或负值。在反向转换过程中,将移除不符合预期正负性的行。

    Attributes:
        int_columns (set): 包含整数值的列名集合。
        float_columns (set): 包含浮点数值的列名集合。
        positive_columns (set): 应包含正值的列名集合。
        negative_columns (set): 应包含负值的列名集合。
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
        
        遍历每一行数据,检查 positive_columns 中是否有负值,
        negative_columns 中是否有正值。如果不符合条件,则丢弃该行。
        """
        logger.info(f"Data reverse-converted by PositiveNegativeFilter Start with Shape: {processed_data.shape}.")
        
        # 创建一个布尔掩码,用于标记需要保留的行
        mask = pd.Series(True, index=processed_data.index)
        
        # 检查 positive_columns
        for col in self.positive_columns:
            if col in processed_data.columns:
                mask &= processed_data[col] >= 0
        
        # 检查 negative_columns
        for col in self.negative_columns:
            if col in processed_data.columns:
                mask &= processed_data[col] <= 0
        
        # 应用掩码筛选数据
        filtered_data = processed_data[mask]
        
        logger.info(f"Data reverse-converted by PositiveNegativeFilter with Output Shape: {filtered_data.shape}.")
        
        return filtered_data

@hookimpl
def register(manager):
    manager.register("PositiveNegativeFilter", PositiveNegativeFilter)