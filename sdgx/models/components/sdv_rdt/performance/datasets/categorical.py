"""Dataset Generators for categorical transformers."""

from abc import ABC

import numpy as np

from sdgx.models.components.sdv_rdt.performance.datasets.base import (
    BaseDatasetGenerator,
)
from sdgx.models.components.sdv_rdt.performance.datasets.datetime import (
    RandomGapDatetimeGenerator,
)
from sdgx.models.components.sdv_rdt.performance.datasets.utils import add_nans


class CategoricalGenerator(BaseDatasetGenerator, ABC):
    """Base class for generators that generate catgorical data."""

    SDTYPE = "categorical"


class RandomIntegerGenerator(CategoricalGenerator):
    """Generator that creates an array of random integers."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        categories = [1, 2, 3, 4, 5]
        return np.random.choice(a=categories, size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 5e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 1000.0,
            },
        }


class RandomIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random integers with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 5e-05, "memory": 1000.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 1000.0,
            },
        }


class RandomStringGenerator(CategoricalGenerator):
    """Generator that creates an array of random strings."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        categories = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
        return np.random.choice(a=categories, size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 500.0},
            "transform": {"time": 1e-05, "memory": 500.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 1000.0,
            },
        }


class RandomStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomStringGenerator.generate(num_rows).astype("O"))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 1e-05, "memory": 1000.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 1000.0,
            },
        }


class RandomMixedGenerator(CategoricalGenerator):
    """Generator that creates an array of random mixed types.

    Mixed types include: int, float, bool, string, datetime.
    """

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        cat_size = 5
        categories = np.hstack(
            [
                cat.astype("O")
                for cat in [
                    RandomGapDatetimeGenerator.generate(cat_size),
                    np.random.randint(0, 100, cat_size),
                    np.random.uniform(0, 100, cat_size),
                    np.arange(cat_size).astype(str),
                    np.array([True, False]),
                ]
            ]
        )

        return np.random.choice(a=categories, size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 1e-05, "memory": 1000.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 2000.0,
            },
        }


class RandomMixedNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of random mixed types with nans.

    Mixed types include: int, float, bool, string, datetime.
    """

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        array = RandomMixedGenerator.generate(num_rows)

        length = len(array)
        num_nulls = np.random.randint(1, length)
        nulls_idx = np.random.choice(range(length), num_nulls)
        nulls = np.random.choice([np.nan, float("nan"), None], num_nulls)
        array[nulls_idx] = nulls

        return array

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 1e-05, "memory": 2000.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 2000.0,
            },
        }


class SingleIntegerGenerator(CategoricalGenerator):
    """Generator that creates an array with a single integer."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        constant = np.random.randint(0, 100)
        return np.full(num_rows, constant)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 3e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 400.0,
            },
        }


class SingleIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array with a single integer with some nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(SingleIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 3e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 500.0,
            },
        }


class SingleStringGenerator(CategoricalGenerator):
    """Generator that creates an array of a single string."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        constant = "A"
        return np.full(num_rows, constant)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 4e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 400.0,
            },
        }


class SingleStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of a single string with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(SingleStringGenerator.generate(num_rows).astype("O"))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 400.0},
            "transform": {"time": 3e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 500.0,
            },
        }


class UniqueIntegerGenerator(CategoricalGenerator):
    """Generator that creates an array of unique integers."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return np.arange(num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 2000.0},
            "transform": {"time": 0.0003, "memory": 500000.0},
            "reverse_transform": {
                "time": 0.0004,
                "memory": 1000000.0,
            },
        }


class UniqueIntegerNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of unique integers with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(UniqueIntegerGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 1000.0},
            "transform": {"time": 0.0002, "memory": 1000000.0},
            "reverse_transform": {
                "time": 0.0002,
                "memory": 1000000.0,
            },
        }


class UniqueStringGenerator(CategoricalGenerator):
    """Generator that creates an array of unique strings."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return np.arange(num_rows).astype(str)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 2000.0},
            "transform": {"time": 0.0002, "memory": 500000.0},
            "reverse_transform": {
                "time": 0.0004,
                "memory": 1000000.0,
            },
        }


class UniqueStringNaNsGenerator(CategoricalGenerator):
    """Generator that creates an array of unique strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(UniqueStringGenerator.generate(num_rows).astype("O"))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-05, "memory": 1000.0},
            "transform": {"time": 0.0005, "memory": 1000000.0},
            "reverse_transform": {
                "time": 0.0002,
                "memory": 1000000.0,
            },
        }
