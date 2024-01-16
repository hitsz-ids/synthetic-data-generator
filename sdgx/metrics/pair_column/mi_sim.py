import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

from sdgx.metrics.pair_column.base import PairMetric
from sdgx.utils import time2int


class MISim(PairMetric):
    """MISim : Mutual Information Similarity

    This class is used to calculate the Mutual Information Similarity between the target columns of real data and synthetic data.

    Currently, we support discrete and continuous(need to be discretized) columns as inputs.
    """

    def __init__(instance) -> None:
        super().__init__()
        instance.lower_bound = 0
        instance.upper_bound = 1
        instance.metric_name = "mutual_information_similarity"
        instance.numerical_bins = 50

    @classmethod
    def calculate(
        cls,
        src_col: pd.Series,
        tar_col: pd.Series,
        metadata: dict,
    ) -> float:
        """
        Calculate the MI similarity for the source data colum and the target data column.
        Args:
            src_data(pd.Series ): the source data column.
            tar_data(pd.Series): the target data column .
            metadata(dict): The metadata that describes the data type of each columns
        Returns:
            MI_similarity (float): The metric value.
        """

        # 传入概率分布数组
        instance = cls()

        col_name = src_col.name
        data_type = metadata[col_name]

        if data_type == "numerical":
            x = np.array(src_col.array)
            src_col = pd.cut(
                x,
                instance.numerical_bins,
                labels=range(instance.numerical_bins),
            )
            x = np.array(tar_col.array)
            tar_col = pd.cut(
                x,
                instance.numerical_bins,
                labels=range(instance.numerical_bins),
            )
            src_col = src_col.to_numpy()
            tar_col = tar_col.to_numpy()

        elif data_type == "category":
            le = LabelEncoder()
            src_list = list(set(src_col.array))
            tar_list = list(set(tar_col.array))
            fit_list = tar_list + src_list
            le.fit(fit_list)

            src_col = le.transform(np.array(src_col.array))
            tar_col = le.transform(np.array(tar_col.array))

        elif data_type == "datetime":
            src_col = src_col.apply(time2int)
            tar_col = tar_col.apply(time2int)
            src_col = pd.cut(
                src_col, bins=instance.numerical_bins, labels=range(instance.numerical_bins)
            )
            tar_col = pd.cut(
                tar_col, bins=instance.numerical_bins, labels=range(instance.numerical_bins)
            )
            src_col = src_col.to_numpy()
            tar_col = tar_col.to_numpy()

        MI_sim = normalized_mutual_info_score(src_col, tar_col)
        return MI_sim

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the MI similarity.
        """
        pass
