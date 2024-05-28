from __future__ import annotations

import pandas as pd

from sdgx.data_processors.base import DataProcessor


class Formatter(DataProcessor):
    """
    Base class for formatters.

    Formatter is used to convert data column from one format to another.

    For example, parse datetime into timestamp when trainning, then format timestamp into datetime when sampling.

    Difference with :ref:`Transformer`:
    - When a single column is used as input, use formatter for formatting issues.
    - Formatter will not add additional columns to the input table.
    - When a whole table is used as input, use :ref:`Transformer`.
    """
