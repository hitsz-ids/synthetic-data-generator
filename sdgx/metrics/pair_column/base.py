import pandas as pd

from sdgx.utils import logger


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
    def check_input(cls, src_col: pd.Series, tar_col: pd.Series, metadata: dict):
        """Input check for table input.
        Args:
            src_data(pd.Series ): the source data column.
            tar_data(pd.Series): the target data column .
            metadata(dict): The metadata that describes the data type of each column
        """
        # Input parameter must not contain None value
        if real_data is None or synthetic_data is None:
            raise TypeError("Input contains None.")
        # check column_names
        tar_name = tar_col.name
        src_name = src_col.name

        # check column_types
        if metadata[tar_name] != metadata[src_name]:
            raise TypeError("Type of Pair is Conflicting.")

        # if type is pd.Series, return directly
        if isinstance(real_data, pd.Series):
            return src_col, tar_col

        # if type is not pd.Series or pd.DataFrame tranfer it to Series
        try:
            src_col = pd.Series(src_col)
            tar_col = pd.Series(tar_col)
            return src_col, tar_col
        except Exception as e:
            logger.error(f"An error occurred while converting to pd.Series: {e}")

        return None, None

    @classmethod
    def calculate(cls, src_col: pd.Series, tar_col: pd.Series, metadata):
        """Calculate the metric value between pair-columns between real table and synthetic table.

        Args:
            src_data(pd.Series ): the source data column.
            tar_data(pd.Series): the target data column .
            metadata(dict): The metadata that describes the data type of each column
        """
        # This method should first check the input
        # such as:
        real_data, synthetic_data = PairMetric.check_input(src_col, tar_col)

        raise NotImplementedError()

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the Mutual Information.
        """
        raise NotImplementedError()

    pass
