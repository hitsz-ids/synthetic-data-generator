# FixedCombinationInspector
from __future__ import annotations

from typing import Any

import pandas as pd

from sdgx.data_models.inspectors.base import Inspector
from sdgx.data_models.inspectors.extension import hookimpl


class FixedCombinationInspector(Inspector):
    """
    FixedCombinationInspector is designed to identify columns in a DataFrame that have fixed relationships based on covariance.

    Attributes:
        fixed_combinations (dict[str, set[str]]): A dictionary mapping column names to sets of column names that have fixed relationships with them.
        _inspect_level (int): The inspection level for this inspector, set to 70.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_combinations: dict[str, set[str]] = {}
        """
        A dictionary mapping column names to sets of column names that have fixed relationships with them.
        """

        self._inspect_level: int = 70
        """
        The inspection level for this inspector, set to 70. This attribute indicates the priority or depth of inspection that this inspector performs relative to other inspectors.
        """

    def fit(self, raw_data: pd.DataFrame, *args, **kwargs):
        """
        Fit the inspector to the raw data.

        处理数值列和字符串列的固定组合关系:
        - 数值列: 通过协方差矩阵计算相关性
        - 字符串列: 通过值的一一对应关系判断
        """
        self.fixed_combinations = {}

        # 1. 处理数值型列
        numeric_columns = raw_data.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_columns) > 0:
            covariance_matrix = raw_data[numeric_columns].dropna().cov()
            for column in covariance_matrix.columns:
                related_columns = set(
                    covariance_matrix.index[covariance_matrix[column].abs() > 0.9]
                )
                related_columns.discard(column)
                if related_columns:
                    self.fixed_combinations[column] = related_columns

        # 2. 处理字符串列
        string_columns = raw_data.columns
        if len(string_columns) > 0:
            # 记录已经找到对应关系的列
            matched_columns = set()

            # 对每列寻找一个对应关系
            for i, col1 in enumerate(string_columns):
                # 如果当前列已经找到对应关系，跳过
                if col1 in matched_columns:
                    continue

                # 只检查当前列之后的列
                for col2 in string_columns[i + 1 :]:
                    if col2 in matched_columns:  # 跳过已匹配的列
                        continue

                    # 检查两列的值是否一一对应
                    pairs = raw_data[[col1, col2]].dropna().drop_duplicates()
                    if (
                        len(pairs) > 0
                        and (pairs.groupby(col1)[col2].nunique() == 1).all()
                        and (pairs.groupby(col2)[col1].nunique() == 1).all()
                    ):
                        # 添加单向的固定关系
                        if col1 not in self.fixed_combinations:
                            self.fixed_combinations[col1] = set()
                        self.fixed_combinations[col1].add(col2)
                        # 标记这两列已经匹配
                        matched_columns.add(col1)
                        matched_columns.add(col2)
                        # 找到一个匹配就跳出内层循环
                        break

        self.ready = True

    def inspect(self, *args, **kwargs) -> dict[str, Any]:
        """Inspect raw data and generate metadata."""

        return {"fixed_combinations": self.fixed_combinations}


@hookimpl
def register(manager):
    manager.register("FixCombinationInspector", FixedCombinationInspector)
