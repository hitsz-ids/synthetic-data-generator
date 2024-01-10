import pandas as pd

from sdgx.log import logger


class PairMetric(object):
    """PairMetric
    Metrics used to evaluate the quality of synthetic data columns.
    """

    upper_bound = None
    lower_bound = None
    metric_name = "Correlation"

    def __init__(self) -> None:
        pass

    @classmethod
    def check_input(
        cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, real_metadata, syn_metadata
    ):
        """Input check for table input.
        Args:
            real_data(pd.DataFrame ): the real (original) data table.
            synthetic_data(pd.DataFrame): the synthetic (generated) data table .
        """
        # Input parameter must not contain None value
        if real_data is None or synthetic_data is None:
            raise TypeError("Input contains None.")
        # check column_names
        real_cols = real_data.columns
        syn_cols = synthetic_data.columns
        if set(real_cols) != set(syn_cols):
            raise TypeError("Columns of Dataframe are Different.")

        # check column_types
        for col in real_cols:
            if real_metadata[col] != syn_metadata[col]:
                raise TypeError("Columns of Dataframe are Different.")

        # if type is pd.DataFrame, return directly
        if isinstance(real_data, pd.DataFrame):
            return real_data, synthetic_data

        # if type is not pd.Series or pd.DataFrame tranfer it to Series
        try:
            real_data = pd.DataFrame(real_data)
            synthetic_data = pd.DataFrame(synthetic_data)
            return real_data, synthetic_data
        except Exception as e:
            logger.error(f"An error occurred while converting to pd.DataFrame: {e}")

        return None, None

    @classmethod
    def calculate(
        cls, real_data: pd.Series | pd.DataFrame, synthetic_data: pd.Series | pd.DataFrame
    ):
        """Calculate the metric value between pair-columns between real table and synthetic table.

        Args:
            real_data(pd.DataFrame or pd.Series): the real (original) data pair.

            synthetic_data(pd.DataFrame or pd.Series): the synthetic (generated) data pair.
        """
        # This method should first check the input
        # such as:
        real_data, synthetic_data = PairMetric.check_input(real_data, synthetic_data)

        raise NotImplementedError()

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the JSD metric.
        """
        raise NotImplementedError()

    pass
