from __future__ import annotations

from sdgx.data_processors.base import DataProcessor


class Filter(DataProcessor):
    """
    Base class for all data filters.

    Filter is a module used to apply rules and remove sampled data that does not conform to the rules.
    """
