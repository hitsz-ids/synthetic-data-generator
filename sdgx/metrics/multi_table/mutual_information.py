import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde

from sdgx.metrics.multi_table.base import MultiTableMetric


class MISim(MultiTableMetric):
    """MISim : Mutual Information Similarity

    This class is used to calculate the Mutual Information Similarity between the target columns of real data and synthetic data.

    Currently, we support discrete and continuous(need to be discretized) columns as inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lower_bound = 0
        self.upper_bound = 1
        self.metric_name = "mutual_information"

    @classmethod
    def calculate(
        cls,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        cols: list[str] | None,
        discrete: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate the JSD value between a real column and a synthetic column.

        Args:
            real_data (pd.DataFrame): The real data.

            synthetic_data (pd.DataFrame): The synthetic data.

            cols (list[str]): The target column to calculat JSD metric.

            discrete (bool): Whether this column is a discrete column.

        Returns:
            MI_similarity (float): The metric value.
        """
        if discrete:
            # 对离散变量求
            MISim.check_input(real_data, synthetic_data)
            joint_pd_real = real_data.groupby(cols, dropna=False).size() / len(real_data)
            joint_pd_syn = synthetic_data.groupby(cols, dropna=False).size() / len(synthetic_data)

            # 对齐操作
            joint_pdf_values_real, joint_pdf_values_syn = joint_pd_real.align(
                joint_pd_syn, fill_value=0
            )
        else:
            # 对连续列
            

        # 传入概率分布数组
        MI_sim = JSD.normailized_mutual_information(joint_pdf_values_real, joint_pdf_values_syn)

        MISim.check_output(MI_sim)

        return MI_sim

    @classmethod
    def check_output(cls, raw_metric_value: float):
        """Check the output value.

        Args:
            raw_metric_value (float):  the calculated raw value of the JSD metric.
        """
        instance = cls()
        if raw_metric_value < instance.lower_bound or raw_metric_value > instance.upper_bound:
            raise ValueError

    @classmethod
    def normailized_mutual_information(cls, p: float, q: float):
        """Calculate the jensen_shannon_divergence of p and q.

        Args:
            p (float): the input parameter p.

            q (float): the input parameter q.
        """
        n_MI = None
        

        return n_MI
