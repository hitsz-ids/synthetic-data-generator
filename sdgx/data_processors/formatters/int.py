from __future__ import annotations

from typing import Any, List

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.formatters.base import Formatter
from sdgx.utils import logger


class IntValueFormatter(Formatter):
    """
    Formatter class for handling Int values in pd.DataFrame.
    """

    int_columns: List = []

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):
        """
        Fit method for the formatter.

        Formatter need to use metadata to record which columns belong to the int type, and convert them back to the int type during post-processing.
        """

        # get from metadata
        self.int_columns = metadata.get("int_columns")

        logger.info("IntValueFormatter Fitted.")
        self.fitted = True

        return

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        No action for convert.
        """

        logger.info("Converting data using IntValueFormatter... Finished  (No Action).")

        return raw_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        """
        reverse_convert method for the formatter.

        Do format conversion for int columns.
        """

        for col in self.int_columns:
            processed_data[col] = processed_data[col].astype(int)

        logger.info("Data reverse-converted by IntValueFormatter.")

        return processed_data


@hookimpl
def register(manager):
    manager.register("IntValueFormatter", IntValueFormatter)
