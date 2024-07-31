from __future__ import annotations

from sdgx.utils import logger


class MultiTableMetric:
    """MultiTableMetric

    Metrics used to evaluate the quality of synthetic multi-table data.
    """

    upper_bound = None
    lower_bound = None
    metric_name = None
    metadata = None
    table_list = []

    def __init__(self, metadata: dict) -> None:
        """Initialization

        Args:
            metadata(dict): This parameter accepts a metadata description dict, which is used to describe the table relations and column description information for each table.
        """
        self.metadata = metadata

    @classmethod
    def check_input(cls, real_data: dict, synthetic_data: dict):
        """Format check for single table input.

        The `real_data` and `synthetic_data` should be dict, which contains tables (in pd.DataFrame).

        Args:
            real_data(dict): the real (original) data table.

            synthetic_data(dict): the synthetic (generated) data table.
        """
        if real_data is None or synthetic_data is None:
            raise TypeError("Input contains None.")

        # The data type should be same
        if type(real_data) is not type(synthetic_data):
            raise TypeError("Data type of real_data and synthetic data should be the same.")

        # if type is dict, return directly
        if (
            isinstance(real_data, dict)
            and len(real_data.keys()) > 0
            and len(synthetic_data.keys()) > 0
        ):
            return real_data, synthetic_data

        logger.error("An error occurred while checking the input.")

        return None, None

    # not a class method
    def calculate(self, real_data: dict, synthetic_data: dict):
        """Calculate the metric value between real tables and synthetic tables.

        Args:

            real_data(dict): the real (original) data table.

            synthetic_data(dict): the synthetic (generated) data table.
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
