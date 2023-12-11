"""
# Formatter: column format conversion tool, the basic description is as follows:
# - For different types of columns, implement the ability to parse, for example: DataTime into timestamp form;
# - for different types of columns, to provide format conversion capabilities
# - Input and output are [column] data.

The difference with Transformer:
# - When a single column is used as input, use formatter for formatting issues.
# - When a whole table is used as input, use data transformer.
# - Usually, in Data Transformer implementations, different formatters are called for different columns.
# - Provide extract method
#extract

"""


class Formatter:
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
