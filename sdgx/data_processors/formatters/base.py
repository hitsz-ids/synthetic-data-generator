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
