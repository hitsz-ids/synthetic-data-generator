from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde

from sdgx.metrics.column.base import ColumnMetric


class JSD(ColumnMetric):
    """JSD : Jensen Shannon Divergence

    This class is used to calculate the Jensen Shannon divergence value betweenthe target columns of real data and synthetic data.

    Currently, we support discrete and continuous columns as inputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lower_bound = 0
        self.upper_bound = 1
        self.metric_name = "jensen_shannon_divergence"

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
            JSD_val (float): The meteic value.
        """
        if discrete:
            # 对离散变量求
            JSD.check_input(real_data, synthetic_data)
            joint_pd_real = real_data.groupby(cols, dropna=False).size() / len(real_data)
            joint_pd_syn = synthetic_data.groupby(cols, dropna=False).size() / len(synthetic_data)

            # 对齐操作
            joint_pdf_values_real, joint_pdf_values_syn = joint_pd_real.align(
                joint_pd_syn, fill_value=0
            )
        else:
            # 对连续列
            # 一个非常大的问题在于求联合概率密度的数组是N^d问题，所以一旦选取3列以上求联合概率密度时间复杂度就不可接受的高，哪怕只取每个值范围只取100个点都算不完
            # 离散列由于是直接用原始数据进行排列求密度，只涉及一次除法，不管多少列都算的很快
            real_data_T = real_data[cols].values.T  # 转置
            syn_data_T = synthetic_data[cols].values.T

            # 对连续列估计KDE概率密度
            kde_joint_real = gaussian_kde(real_data_T)
            kde_joint_syn = gaussian_kde(syn_data_T)

            # 均匀取点，取值范围选取真实数据集的最大最小范围
            variables_range = [np.linspace(min(col), max(col), 100) for col in real_data_T]
            grid_points = np.meshgrid(*variables_range)
            grid_points_flat = np.vstack([item.ravel() for item in grid_points])

            # 计算概率分布数组
            joint_pdf_values_real = (
                kde_joint_real(grid_points_flat).reshape(grid_points[0].shape).ravel()
            )
            joint_pdf_values_syn = (
                kde_joint_syn(grid_points_flat).reshape(grid_points[0].shape).ravel()
            )

        # 传入概率分布数组
        JSD_val = JSD.jensen_shannon_divergence(joint_pdf_values_real, joint_pdf_values_syn)

        JSD.check_output(JSD_val)

        return JSD_val

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
    def jensen_shannon_divergence(cls, p: float, q: float):
        """Calculate the jensen_shannon_divergence of p and q.

        Args:
            p (float): the input parameter p.

            q (float): the input parameter q.
        """
        # Calculate the average distribution of p and q
        m = 0.5 * (p + q)

        # Calculate KL divergence
        kl_p = entropy(p, m, base=2)
        kl_q = entropy(q, m, base=2)

        # Calculate Jensen Shannon divergence
        js_divergence = 0.5 * (kl_p + kl_q)

        return js_divergence
