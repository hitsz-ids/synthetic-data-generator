import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.cluster import normalized_mutual_info_score

from sdgx.metrics.multi_table.base import MultiTableMetric


def Jaccard_index(A, B):
    return min(A, B) / max(A, B)


def time2int(datetime, form):
    time_array = time.strptime(datetime, form)
    time_stamp = int(time.mktime(time_array))
    return time_stamp


class MISim(MultiTableMetric):
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
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata:dict
    ) -> pd.DataFrame:
        """
        Calculate the JSD value between a real column and a synthetic column.
        Args:
            real_data (pd.DataFrame): The real data.

            synthetic_data (pd.DataFrame): The synthetic data.

        Returns:
            MI_similarity (float): The metric value.
        """

        # 传入概率分布数组

        columns = synthetic_data.columns
        n = len(columns)

        for col in columns:
            data_type = metadata[col]
            if data_type == "numerical":
                # max_value = real_data[col].max()
                # min_value = real_data[col].min()
                real_data[col] = pd.cut(a, self.numerical_bins, labels=range(self.numerical_bins))
                synthetic_data[col] = pd.cut(
                    a, self.numerical_bins, labels=range(self.numerical_bins)
                )

            elif data_type == "datetime":
                real_data[col] = real_data[col].apply(time2int)
                synthetic_data[col] = synthetic_data[col].apply(time2int)
                real_data[col] = pd.cut(a, self.numerical_bins, labels=range(self.numerical_bins))
                synthetic_data[col] = pd.cut(
                    a, self.numerical_bins, labels=range(self.numerical_bins)
                )

        nMI_sim = np.zeros((n, n))

        for i in range(len(columns)):
            for j in range(len(columns)):
                syn_MI_ij = normalized_mutual_info_score(
                    synthetic_data[columns[i]], synthetic_data[columns[j]]
                )
                real_MI_ij = normalized_mutual_info_score(
                    real_data[columns[i]], real_data[columns[j]]
                )
                nMI_sim[i][j] = Jaccard_index(syn_MI_ij, real_MI_ij)

        MI_sim = np.sum(nMI_sim) / n / n
        # test
        MISim.check_output(MI_sim)

        return MI_sim

    @classmethod
    def check_output(raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the JSD metric.
        """
        # instance = cls()
        if raw_metric_value < self.lower_bound or raw_metric_value > self.upper_bound:
            raise ValueError

    # @classmethod
    # def normailized_mutual_information(cls, p: float, q: float):
    #     """Calculate the jensen_shannon_divergence of p and q.

    #     Args:
    #         p (float): the input parameter p.

    #         q (float): the input parameter q.
    #     """
    #     n_MI = None

    #     return n_MI
