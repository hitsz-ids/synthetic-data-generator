from sdgx.data_processors.base import DataProcessor


class Generator(DataProcessor):
    """
    Base class for generators.

    Generator is used to generate data based on relationships, rules and restrictions defined by metadata.

    For example, generate random numbers between 0 and 1 as a fraction.
    """
