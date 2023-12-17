"""Dataset Generators for datetime transformers."""

import datetime
from abc import ABC

import numpy as np
import pandas as pd

from sdgx.models.components.sdv_rdt.performance.datasets.base import (
    BaseDatasetGenerator,
)
from sdgx.models.components.sdv_rdt.performance.datasets.utils import add_nans


class DatetimeGenerator(BaseDatasetGenerator, ABC):
    """Base class for generators that generate datatime data."""

    SDTYPE = "datetime"


class RandomGapDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps between them."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        today = datetime.datetime.today()
        delta = datetime.timedelta(days=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype="datetime64")

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 5e-05, "memory": 500.0},
            "transform": {"time": 5e-05, "memory": 300.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 1000.0,
            },
        }


class RandomGapSecondsDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps of seconds between them."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        today = datetime.datetime.today()
        delta = datetime.timedelta(seconds=1)
        dates = [(np.random.random() * delta + today) for i in range(num_rows)]
        return np.array(dates, dtype="datetime64")

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 5e-05, "memory": 500.0},
            "transform": {"time": 5e-05, "memory": 300.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 1000.0,
            },
        }


class RandomGapDatetimeNaNsGenerator(DatetimeGenerator):
    """Generator that creates dates with random gaps and NaNs."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        dates = RandomGapDatetimeGenerator.generate(num_rows)
        return add_nans(dates.astype("O"))

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 5e-05, "memory": 500.0},
            "transform": {"time": 5e-05, "memory": 1000.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 1000.0,
            },
        }


class EqualGapHoursDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with hour gaps between them."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        today = datetime.datetime.today()
        delta = datetime.timedelta
        dates = [delta(hours=i) + today for i in range(num_rows)]
        return np.array(dates, dtype="datetime64")

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 5e-05, "memory": 500.0},
            "transform": {"time": 5e-05, "memory": 300.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 1000.0,
            },
        }


class EqualGapDaysDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with 1 day gaps between them."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        today = datetime.datetime.today()
        delta = datetime.timedelta

        today = min(datetime.datetime.today(), pd.Timestamp.max - delta(num_rows))
        dates = [delta(i) + today for i in range(num_rows)]

        return np.array(dates, dtype="datetime64")

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 5e-05, "memory": 500.0},
            "transform": {"time": 5e-05, "memory": 300.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 1000.0,
            },
        }


class EqualGapWeeksDatetimeGenerator(DatetimeGenerator):
    """Generator that creates dates with 1 week gaps between them."""

    @staticmethod
    def generate(num_rows):
        """Generate a ``num_rows`` number of rows."""
        today = datetime.datetime.today()
        delta = datetime.timedelta

        today = datetime.datetime.today()
        dates = [min(delta(weeks=i) + today, pd.Timestamp.max) for i in range(num_rows)]

        return np.array(dates, dtype="datetime64")

    @staticmethod
    def get_performance_thresholds():
        """Return the expected threseholds."""
        return {
            "fit": {"time": 5e-05, "memory": 500.0},
            "transform": {"time": 5e-05, "memory": 300.0},
            "reverse_transform": {
                "time": 5e-05,
                "memory": 1000.0,
            },
        }
