from __future__ import annotations

from typing import Any

from pandas import DataFrame

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.transformers.base import Transformer
from sdgx.utils import logger


class NonValueTransformer(Transformer):
    """
    Transformer class for handling missing values in data.

    This Transformer is mainly used as a reference for Transformer to facilitate developers to quickly understand the role of Transformer.
    """

    fill_na_value = 0

    drop_na = False

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
