"""Dataset Generators for Personal Identifiable Information transformers."""

from abc import ABC

import numpy as np

from sdgx.models.components.sdv_rdt.performance.datasets.base import (
    BaseDatasetGenerator,
)
from sdgx.models.components.sdv_rdt.performance.datasets.utils import add_nans


class PIIGenerator(BaseDatasetGenerator, ABC):
    """Base class for generators that generate PII data."""

    SDTYPE = "pii"


class RandomStringGenerator(PIIGenerator):
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
            "fit": {"time": 1e-05, "memory": 500.0},
            "transform": {"time": 1e-05, "memory": 500.0},
            "reverse_transform": {
                "time": 2e-05,
                "memory": 1000.0,
            },
        }


class RandomStringNaNsGenerator(PIIGenerator):
    """Generator that creates an array of random strings with nans."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        return add_nans(RandomStringGenerator.generate(num_rows).astype("O"))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 1e-05, "memory": 400.0},
            "transform": {"time": 1e-05, "memory": 1000.0},
            "reverse_transform": {
                "time": 2e-05,
                "memory": 1000.0,
            },
        }
