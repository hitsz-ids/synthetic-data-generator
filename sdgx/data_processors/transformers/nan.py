from __future__ import annotations

from typing import Any
from pandas import DataFrame

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class NonValueTransformer(Transformer):
    """
    A transformer class for handling missing values in a DataFrame.

    This class provides functionality to either drop rows with missing values or fill them with a specified value.

    Attributes:
        fill_na_value (int): The value to fill missing values in the data.
        drop_na (bool): A boolean flag indicating whether to drop rows with missing values or fill them with `fill_na_value`.

    Methods:
        fit(metadata: Metadata | None = None, **kwargs: dict[str, Any]): Fit method for the transformer.
        convert(raw_data: DataFrame) -> DataFrame: Convert method to handle missing values in the input data.
        reverse_convert(processed_data: DataFrame) -> DataFrame: Reverse_convert method for the transformer.
    """

    fill_na_value = 0
    """
    The value to fill missing values in the data.

    If `drop_na` is set to `False`, this value will be used to fill missing values in the data.
    """

    drop_na = True
    """
    A boolean flag indicating whether to drop rows with missing values or fill them with `fill_na_value`.

    If `True`, rows with missing values will be dropped. If `False`, missing values will be filled with `fill_na_value`.
    """

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the transformer.

        Does not require any action.
        """
        logger.info("NonValueTransformer Fitted.")

        self.fitted = True

        return

    def convert(self, raw_data: DataFrame) -> DataFrame:
        """
        Convert method to handle missing values in the input data.
        """

        logger.info("Converting data using NonValueTransformer...")

        if self.drop_na:
            res = raw_data.dropna()
        else:
            res = raw_data.fillna(value=self.fill_na_value)

        logger.info("Converting data using NonValueTransformer... Finished.")

        return res

    def reverse_convert(self, processed_data: DataFrame) -> DataFrame:
        """
        Reverse_convert method for the transformer.

        Does not require any action.
        """
        logger.info("Data reverse-converted by NonValueTransformer (No Action).")

        return processed_data

    pass


@hookimpl
def register(manager):
    manager.register("NonValueTransformer", NonValueTransformer)
