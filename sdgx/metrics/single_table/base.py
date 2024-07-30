import pandas as pd

from sdgx.utils import logger


class SingleTableMetric:
    """SingleTableMetric

    Metrics used to evaluate the quality of single table synthetic data.
    """

    upper_bound = None
    lower_bound = None
    metric_name = None
    metadata = None

    def __init__(self, metadata: dict) -> None:
        """Initialization

        Args:
            metadata(dict): This parameter accepts a metadata description dict, which is used to describe the column description information of the table.
        """
        self.metadata = metadata
        pass

    @classmethod
    def check_input(cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Input check for single table input.

        Args:
            real_data(pd.DataFrame): the real (original) data table.

            synthetic_data(pd.DataFrame): the synthetic (generated) data table.
        """
        # should be pd.DataFrame
        # Input parameter must not contain None value
        if real_data is None or synthetic_data is None:
            raise TypeError("Input contains None.")

        # The data type should be same
        if type(real_data) is not type(synthetic_data):
            raise TypeError("Data type of real_data and synthetic data should be the same.")

        # if type is pd.Series, return directly
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

    def calculate(cls, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """Calculate the metric value between a real table and a synthetic table.

        Args:
            real_data(pd.DataFrame): the real (original) data table.

            synthetic_data(pd.DataFrame): the synthetic (generated) data table.
        """

        raise NotImplementedError()

    @classmethod
    def check_output(raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the Mutual Information Similarity.
        """
        raise NotImplementedError()

    pass
