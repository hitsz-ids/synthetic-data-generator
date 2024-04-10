from __future__ import annotations

import pandas as pd

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.base import DataProcessor
from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader


class Transformer(DataProcessor):
    """
    Base class for transformers.

    Transformer is used to transform table data from one format to another.
    For example, encode discrete column into one hot encoding.

    To achieve that, Transformer can use :ref:`Formatter` and :ref:`Inspector` to help.
    """

    def fit(self, metadata: Metadata | None = None, tabular_data: DataLoader | pd.DataFrame = None):
        """
        Fit method for the transformer.
        """

        return
