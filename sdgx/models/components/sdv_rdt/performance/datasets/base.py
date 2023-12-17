"""Base class for all the Dataset Generators."""

from abc import ABC, abstractmethod


class BaseDatasetGenerator(ABC):
    """Parent class for all the Dataset Generators."""

    SDTYPE = None

    @staticmethod
    @abstractmethod
    def generate(num_rows):
        """Return array of data. This method serves as a template for dataset generators.

        Args:
            num_rows (int):
                Number of rows to generate.

        Returns:
            numpy.ndarray of size ``num_rows``
        """
        raise NotImplementedError()

    @classmethod
    def get_subclasses(cls):
        """Recursively find subclasses of this Baseline.

        Returns:
            list:
                List of all subclasses of this class.
        """
        subclasses = []
        for subclass in cls.__subclasses__():
            if ABC not in subclass.__bases__:
                subclasses.append(subclass)

            subclasses += subclass.get_subclasses()

        return subclasses

    @staticmethod
    @abstractmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        raise NotImplementedError()
