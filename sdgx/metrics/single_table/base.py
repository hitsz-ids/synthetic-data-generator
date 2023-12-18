import pandas as pd


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

        raise NotImplementedError()

    def calculate(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
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
            raw_metric_value (float):  the calculated raw value of the JSD metric.
        """
        raise NotImplementedError()

    pass
