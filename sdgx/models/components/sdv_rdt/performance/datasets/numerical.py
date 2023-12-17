"""Dataset Generators for numerical transformers."""

from abc import ABC

import numpy as np

from sdgx.models.components.sdv_rdt.performance.datasets.base import (
    BaseDatasetGenerator,
)
from sdgx.models.components.sdv_rdt.performance.datasets.utils import add_nans


class NumericalGenerator(BaseDatasetGenerator, ABC):
    """Base class for generators that create numerical data."""

    SDTYPE = "numerical"


class RandomIntegerGenerator(NumericalGenerator):
    """Generator that creates an array of random integers."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        ii32 = np.iinfo(np.int32)
        return np.random.randint(ii32.min, ii32.max, num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 5e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 400.0,
            },
        }


class RandomIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of random integers with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 4e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 2e-05,
                "memory": 300.0,
            },
        }


class ConstantIntegerGenerator(NumericalGenerator):
    """Generator that creates a constant array with a random integer."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        ii32 = np.iinfo(np.int32)
        constant = np.random.randint(ii32.min, ii32.max)
        return np.full(num_rows, constant)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 400.0},
            "transform": {"time": 1e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 400.0,
            },
        }


class ConstantIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates a constant array with a random integer with some nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(ConstantIntegerGenerator.generate(num_rows).astype(float))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 600.0},
            "transform": {"time": 3e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 2e-05,
                "memory": 300.0,
            },
        }


class AlmostConstantIntegerGenerator(NumericalGenerator):
    """Generator that creates an array with 2 only values, one of them repeated."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1])
        array = np.concatenate([values, additional_values])
        np.random.shuffle(array)
        return array

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 1e-05, "memory": 2000.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 2000.0,
            },
        }


class AlmostConstantIntegerNaNsGenerator(NumericalGenerator):
    """Generator that creates an array with 2 only values, one of them repeated, and NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        ii32 = np.iinfo(np.int32)
        values = np.random.randint(ii32.min, ii32.max, size=2)
        additional_values = np.full(num_rows - 2, values[1]).astype(float)
        array = np.concatenate([values, add_nans(additional_values)])
        np.random.shuffle(array)
        return array

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 3e-05, "memory": 1000.0},
            "reverse_transform": {
                "time": 2e-05,
                "memory": 1000.0,
            },
        }


class NormalGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return np.random.normal(size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 1e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 1e-05,
                "memory": 400.0,
            },
        }


class NormalNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(NormalGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 4e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 300.0,
            },
        }


class BigNormalGenerator(NumericalGenerator):
    """Generator that creates an array of big normally distributed float values."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return np.random.normal(scale=1e10, size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 5e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 400.0,
            },
        }


class BigNormalNaNsGenerator(NumericalGenerator):
    """Generator that creates an array of normally distributed float values, with NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(BigNormalGenerator.generate(num_rows))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-03, "memory": 2500.0},
            "transform": {"time": 3e-05, "memory": 400.0},
            "reverse_transform": {
                "time": 2e-05,
                "memory": 300.0,
            },
        }
