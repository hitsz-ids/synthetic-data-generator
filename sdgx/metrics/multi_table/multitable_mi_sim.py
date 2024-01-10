import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.cluster import normalized_mutual_info_score

from sdgx.metrics.multi_table.base import MultiTableMetric
from sdgx.metrics.pair_column.mi_sim import MISim





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
        real_data: pd.DataFrame, synthetic_data: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        Calculate the JSD value between a real column and a synthetic column.
        Args:
            real_data (pd.DataFrame): The real data.
            synthetic_data (pd.DataFrame): The synthetic data.
            metadata(dict): The metadata that describes the data type of each column

        Returns:
            MI_similarity (float): The metric value.
        """

        # 传入概率分布数组

        columns = synthetic_data.columns
        n = len(columns)
        mi_sim_instance = MISim()
        nMI_sim = np.zeros((n, n))

        for i in range(len(columns)):
            for j in range(len(columns)):
                syn_data = pd.concat([synthetic_data[columns[i]], synthetic_data[columns[j]]], axis=1)
                real_data = pd.concat([real_data[columns[i]], real_data[columns[j]]], axis=1)

                nMI_sim[i][j] = mi_sim_instance.calculate(
                    real_data, syn_data ,metadata
                )

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
