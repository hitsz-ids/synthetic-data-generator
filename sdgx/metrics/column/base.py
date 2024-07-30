from __future__ import annotations

import pandas as pd

from sdgx.utils import logger


class ColumnMetric(object):
    """ColumnMetric

    Metrics used to evaluate the quality of synthetic data columns.
    """

    upper_bound = None
    lower_bound = None
    metric_name = "Accuracy"

    def __init__(self) -> None:
        pass

    @classmethod
    def check_input(
        cls, real_data: pd.Series | pd.DataFrame, synthetic_data: pd.Series | pd.DataFrame
    ):
        """Input check for column or table input.

        Args:
            real_data(pd.DataFrame or pd.Series): the real (original) data table / column.

            synthetic_data(pd.DataFrame or pd.Series): the synthetic (generated) data table / column.
        """

        # Input parameter must not contain None value
        if real_data is None or synthetic_data is None:
            raise TypeError("Input contains None.")

        # The data type should be same
        if type(real_data) is not type(synthetic_data):
            raise TypeError("Data type of real_data and synthetic data should be the same.")

        # Check some data-types that must not be allowed
        if type(real_data) in [int, float, str]:
            raise TypeError("real_data's type must not be None, int, float or str")

        # if type is pd.Series, return directly
        if isinstance(real_data, pd.Series) or isinstance(real_data, pd.DataFrame):
            return real_data, synthetic_data

        # if type is not pd.Series or pd.DataFrame tranfer it to Series
        try:
            real_data = pd.Series(real_data)
            synthetic_data = pd.Series(synthetic_data)
            return real_data, synthetic_data
        except Exception as e:
            logger.error(f"An error occurred while converting to pd.Series: {e}")

        return None, None

    @classmethod
    def calculate(
        cls, real_data: pd.Series | pd.DataFrame, synthetic_data: pd.Series | pd.DataFrame
    ):
        """Calculate the metric value between columns between real table and synthetic table.
        Args:
            real_data(pd.DataFrame or pd.Series): the real (original) data table / column.
            synthetic_data(pd.DataFrame or pd.Series): the synthetic (generated) data table / column.
        """
        # This method should first check the input
        # such as:
        real_data, synthetic_data = ColumnMetric.check_input(real_data, synthetic_data)

        raise NotImplementedError()

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the JSD metric.
        """
        raise NotImplementedError()

    pass
