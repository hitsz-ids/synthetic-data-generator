from __future__ import annotations

from typing import Any

from pandas import DataFrame

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class NonValueTransformer(Transformer):
    """
    A transformer class designed to handle missing values in a DataFrame. It can either drop rows with missing values or fill them with specified values.

    Attributes:
        int_columns (set): A set of column names that contain integer values.
        float_columns (set): A set of column names that contain float values.
        column_list (list): A list of all column names in the DataFrame.
        fill_na_value_int (int): The value to fill missing integer values with. Default is 0.
        fill_na_value_float (float): The value to fill missing float values with. Default is 0.0.
        fill_na_value_default (str): The value to fill missing values for non-numeric columns with. Default is 'NAN_VALUE'.
        drop_na (bool): A flag indicating whether to drop rows with missing values. If True, rows with missing values are dropped. If False, missing values are filled with specified values. Default is False.
    """

    int_columns: set
    """
    A set of column names that contain integer values.
    """

    float_columns: set
    """
    A set of column names that contain float values.
    """

    column_list: list
    """
    A list of all column names in the DataFrame.
    """

    fill_na_value_int: int
    """
    The value to fill missing integer values with. Default is 0.
    """

    fill_na_value_float: float
    """
    The value to fill missing float values with. Default is 0.0.
    """

    fill_na_value_default: str
    """
    The value to fill missing values for non-numeric columns with. Default is 'NAN_VALUE'.
    """

    drop_na: bool
    """
    A boolean flag indicating whether to drop rows with missing values or fill them with `fill_na_value`.

    If `True`, rows with missing values will be dropped.
    If `False`, missing values will be filled with `fill_na_value`.

    Currently, the default setting is False, which means rows with missing values are not dropped.
    """

    def __init__(self):
        self.int_columns = set()
        self.float_columns = set()
        self.column_list = []
        self.fill_na_value_int = 0
        self.fill_na_value_float = 0.0
        self.fill_na_value_default = "NAN_VALUE"
        self.drop_na = False

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.
        """

        logger.info("NonValueTransformer Fitted.")

        for key, value in kwargs.items():
            if key == "drop_na":
                if not isinstance(value, str):
                    raise ValueError("fill_na_value must be of type <str>")
                self.drop_na = value

        # record numeric columns
        # int columns
        for each_col in metadata.int_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == "int":
                self.int_columns.add(each_col)

        logger.info(f"NonValueTransformer get int columns: {self.int_columns}.")

        # float columns
        for each_col in metadata.float_columns:
            if each_col not in metadata.column_list:
                continue
            if metadata.get_column_data_type(each_col) == "float":
                self.float_columns.add(each_col)

        logger.info(f"NonValueTransformer get float columns: {self.float_columns}.")

        # get all column list
        self.column_list = metadata.column_list

        logger.info(f"NonValueTransformer get column list from metadata: {self.column_list}.")

        self.fitted = True

    def convert(self, raw_data: DataFrame) -> DataFrame:
        """
        Convert method to handle missing values in the input data.
        """

        logger.info("Converting data using NonValueTransformer...")

        if self.drop_na:
            logger.info("Converting data using NonValueTransformer... Finished (Drop NA).")
            return raw_data.dropna()

        res = raw_data

        # fill numeric nan value
        for each_col in self.int_columns:
            res[each_col] = res[each_col].fillna(self.fill_na_value_int)
        for each_col in self.float_columns:
            res[each_col] = res[each_col].fillna(self.fill_na_value_float)

        # fill other non-numeric nan value
        for each_col in self.column_list:
            if each_col in self.int_columns or each_col in self.float_columns:
                continue
            res[each_col] = res[each_col].fillna(self.fill_na_value_default)

        logger.info("Converting data using NonValueTransformer... Finished.")

        return res

    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        """
        Reverse_convert method for the transformer.

        Does not require any action.
        """

        def replace_nan_value(df):
            """
            Scans all rows and columns in the DataFrame and replaces all cells with the value "NAN_VALUE", which is self.fill_na_value_default, with an empty string.

            Parameters:
            df (pd.DataFrame): The input DataFrame.

            Returns:
            pd.DataFrame: The DataFrame after replacement.
            """
            # Use the replace method of DataFrame to replace "NAN_VALUE" with an empty string
            df_replaced = df.replace(self.fill_na_value_default, "")
            return df_replaced

        logger.info("Data reverse-converted by NonValueTransformer.")

        return replace_nan_value(processed_data)

    pass


@hookimpl
def register(manager):
    manager.register("NonValueTransformer", NonValueTransformer)
