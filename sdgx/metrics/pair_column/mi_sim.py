import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.cluster import normalized_mutual_info_score

from sdgx.metrics.pair_column.base import PairMetric
from sdgx.utils import time2int


class MISim(PairMetric):
    """MISim : Mutual Information Similarity

    This class is used to calculate the Mutual Information Similarity between the target columns of real data and synthetic data.

    Currently, we support discrete and continuous(need to be discretized) columns as inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lower_bound = 0
        self.upper_bound = 1
        self.metric_name = "mutual_information_similarity"
        self.numerical_bins = 50

    @classmethod
    def calculate(
        src_col: pd.DataFrame,
        tar_col: pd.DataFrame,
        metadata: dict,
    ) -> float:
        """
        Calculate the JSD value between a real column and a synthetic column.
        Args:
            src_data(pd.Series ): the source data column.
            tar_data(pd.Series): the target data column .
            metadata(dict): The metadata that describes the data type of each columns
        Returns:
            MI_similarity (float): The metric value.
        """

        # 传入概率分布数组

        col_name = src_col.name
        data_type = metadata[col_name]
        if data_type == "numerical":
            src_col = pd.cut(src_col, self.numerical_bins, labels=range(self.numerical_bins))
            tar_col = pd.cut(tar_col, self.numerical_bins, labels=range(self.numerical_bins))

        elif data_type == "datetime":
            src_col = src_col.apply(time2int)
            tar_col = tar_col.apply(time2int)
            src_col = pd.cut(src_col, self.numerical_bins, labels=range(self.numerical_bins))
            tar_col = pd.cut(tar_col, self.numerical_bins, labels=range(self.numerical_bins))

        MI_sim = normalized_mutual_info_score(src_col, tar_col)

        return MI_sim

    @classmethod
    def check_output(raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the MI similarity.
        """
        pass
