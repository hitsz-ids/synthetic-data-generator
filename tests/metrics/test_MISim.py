from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from sdgx.metrics.pair_column.mi_sim import MISim

# 创建测试数据


@pytest.fixture
def test_data_category():
    role_set = ["admin", "user", "guest"]
    df = pd.DataFrame(
        {
            "role1": [random.choice(role_set) for _ in range(10)],
            "role2": [random.choice(role_set) for _ in range(10)],
        }
    )
    return df


@pytest.fixture
def test_data_num():
    df = pd.DataFrame(
        {
            "feature_x": [random.random() for _ in range(10)],
            "feature_y": [random.random() for _ in range(10)],
        }
    )
    return df


@pytest.fixture
def mi_sim_instance():
    return MISim()


def test_MISim_discrete(test_data_category, mi_sim_instance):
    metadata = {"role1": "category", "role2": "category"}
    col_src = "role1"
    col_tar = "role2"
    result = mi_sim_instance.calculate(
        test_data_category[col_src], test_data_category[col_tar], metadata
    )
    result1 = mi_sim_instance.calculate(
        test_data_category[col_src], test_data_category[col_src], metadata
    )
    result2 = mi_sim_instance.calculate(
        test_data_category[col_tar], test_data_category[col_src], metadata
    )

    assert result >= 0
    assert result <= 1
    assert result1 == 1
    assert result2 == result


def test_MISim_continuous(test_data_num, mi_sim_instance):
    metadata = {"feature_x": "numerical", "feature_y": "numerical"}
    col_src = "feature_x"
    col_tar = "feature_y"
    result = mi_sim_instance.calculate(
        test_data_category[col_src], test_data_category[col_tar], metadata
    )
    result1 = mi_sim_instance.calculate(
        test_data_category[col_src], test_data_category[col_src], metadata
    )
    result2 = mi_sim_instance.calculate(
        test_data_category[col_tar], test_data_category[col_src], metadata
    )

    assert result >= 0
    assert result <= 1
    assert result1 == 1
    assert result2 == result


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
