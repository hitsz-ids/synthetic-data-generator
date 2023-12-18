"""Dataset Generators for boolean transformers."""

from abc import ABC

import numpy as np

from sdgx.models.components.sdv_rdt.performance.datasets.base import (
    BaseDatasetGenerator,
)

MAX_PERCENT_NULL = 50  # cap the percentage of null values at 50%
MIN_PERCENT = 20  # the minimum percentage of true or false is 20%


class BooleanGenerator(BaseDatasetGenerator, ABC):
    """Base class for generators that generate boolean data."""

    SDTYPE = "boolean"


class RandomBooleanGenerator(BooleanGenerator):
    """Generator that creates dataset of random booleans."""

    @staticmethod
    def generate(num_rows):
        """Generate an array of random booleans.

        Args:
            num_rows (int):
                Number of rows of booleans to generate.

        Returns:
            numpy.ndarray of size ``num_rows`` containing random booleans.
        """
        return np.random.choice(a=[True, False], size=num_rows)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-5, "memory": 400.0},
            "transform": {"time": 1e-5, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-5,
                "memory": 500.0,
            },
        }


class RandomBooleanNaNsGenerator(BooleanGenerator):
    """Generator that creates an array of random booleans with nulls."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        percent_null = np.random.randint(MIN_PERCENT, MAX_PERCENT_NULL)
        percent_true = (100 - percent_null) / 2
        percent_false = 100 - percent_true - percent_null

        return np.random.choice(
            a=[True, False, None],
            size=num_rows,
            p=[percent_true / 100, percent_false / 100, percent_null / 100],
        )

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 2e-5, "memory": 400.0},
            "transform": {"time": 1e-5, "memory": 1000.0},
            "reverse_transform": {
                "time": 5e-5,
                "memory": 1000.0,
            },
        }


class RandomSkewedBooleanGenerator(BooleanGenerator):
    """Generator that creates dataset of random booleans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        percent_true = np.random.randint(MIN_PERCENT, 100 - MIN_PERCENT)

        return np.random.choice(
            a=[True, False],
            size=num_rows,
            p=[percent_true / 100, (100 - percent_true) / 100],
        )

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-5, "memory": 400.0},
            "transform": {"time": 1e-5, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-5,
                "memory": 500.0,
            },
        }


class RandomSkewedBooleanNaNsGenerator(BooleanGenerator):
    """Generator that creates an array of random booleans with nulls."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        percent_null = np.random.randint(MIN_PERCENT, MAX_PERCENT_NULL)
        percent_true = np.random.randint(MIN_PERCENT, 100 - percent_null - MIN_PERCENT)
        percent_false = 100 - percent_null - percent_true

        return np.random.choice(
            a=[True, False, None],
            size=num_rows,
            p=[percent_true / 100, percent_false / 100, percent_null / 100],
        )

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-5, "memory": 400.0},
            "transform": {"time": 1e-5, "memory": 1000.0},
            "reverse_transform": {
                "time": 5e-5,
                "memory": 1000.0,
            },
        }


class ConstantBooleanGenerator(BooleanGenerator):
    """Generator that creates a constant array with either True or False."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        constant = np.random.choice([True, False])
        return np.full(num_rows, constant)

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-5, "memory": 400.0},
            "transform": {"time": 1e-5, "memory": 400.0},
            "reverse_transform": {
                "time": 5e-5,
                "memory": 500.0,
            },
        }


class ConstantBooleanNaNsGenerator(BooleanGenerator):
    """Generator that creates a constant array with either True or False with some nulls."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        constant = np.random.choice([True, False])
        percent_null = np.random.randint(MIN_PERCENT, MAX_PERCENT_NULL)

        return np.random.choice(
            a=[constant, None],
            size=num_rows,
            p=[(100 - percent_null) / 100, percent_null / 100],
        )

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-5, "memory": 400.0},
            "transform": {"time": 1e-5, "memory": 1000.0},
            "reverse_transform": {
                "time": 5e-5,
                "memory": 1000.0,
            },
        }
