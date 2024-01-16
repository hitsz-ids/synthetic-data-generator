from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from sdgx.metrics.pair_column.mi_sim import MISim


# 创建测试数据
@pytest.fixture
def dummy_data_cate(dummy_single_table_path):
    df = pd.read_csv(dummy_single_table_path)
    yield df["role"]


@pytest.fixture
def dummy_data_num(dummy_single_table_path):
    df = pd.read_csv(dummy_single_table_path)
    yield df["feature_x"]


@pytest.fixture
def test_data_category():
    role_set = ["admin", "user", "guest"]
    df = pd.Series(
        {
            "role": [random.choice(role_set) for _ in range(10)],
        }
    )
    return df


@pytest.fixture
def test_data_num():
    df = pd.Series(
        {
            "feature_x": [random.random() for _ in range(10)],
        }
    )
    return df


@pytest.fixture
def mi_sim_instance():
    return MISim()


def test_MISim_discrete(dummy_data_cate, test_data_category):
    metadata = {"role": "category"}
    result = mi_sim_instance.calculate(dummy_data_cate, test_data_category, metadata)
    result1 = mi_sim_instance.calculate(dummy_data_cate, dummy_data_cate, metadata)
    result2 = mi_sim_instance.calculate(test_data_category, dummy_data_cate, metadata)

    assert result >= 0
    assert result <= 1
    assert result1 == 1
    assert result2 == result


def test_MISim_continuous(dummy_data_num, test_data_num):
    metadata = {"feature_x": "numerical"}
    result = mi_sim_instance.calculate(dummy_data_num, test_data_num, metadata)
    result1 = mi_sim_instance.calculate(dummy_data_num, dummy_data_num, metadata)
    result2 = mi_sim_instance.calculate(test_data_num, dummy_data_num, metadata)

    assert result >= 0
    assert result <= 1
    assert result1 == 1
    assert result2 == result
