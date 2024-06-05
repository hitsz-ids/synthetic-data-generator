from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from sdgx.metrics.pair_column.mi_sim import MISim

# 创建测试数据


def generate_random_time(start_date, end_date):
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    random_time_delta = random.randint(0, int((end_datetime - start_datetime).total_seconds()))
    random_datetime = start_datetime + timedelta(seconds=random_time_delta)

    return random_datetime


@pytest.fixture
def test_data_time():
    start_date = "1900-01-01"
    end_date = "2023-12-31"
    df = pd.DataFrame(
        {
            "time_x": [generate_random_time(start_date, end_date) for _ in range(10)],
            "time_y": [generate_random_time(start_date, end_date) for _ in range(10)],
        }
    )
    return df


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
    assert round(result2, 9) == round(result, 9)


def test_MISim_continuous(test_data_num, mi_sim_instance):
    metadata = {"feature_x": "numerical", "feature_y": "numerical"}
    col_src = "feature_x"
    col_tar = "feature_y"
    result = mi_sim_instance.calculate(test_data_num[col_src], test_data_num[col_tar], metadata)
    result1 = mi_sim_instance.calculate(test_data_num[col_src], test_data_num[col_src], metadata)
    result2 = mi_sim_instance.calculate(test_data_num[col_tar], test_data_num[col_src], metadata)

    assert result >= 0
    assert result <= 1
    assert np.isclose(result1, 1, atol=1e-9)
    assert np.isclose(result, result2, atol=1e-9)


def test_MISim_time(test_data_time, mi_sim_instance):
    metadata = {"time_x": "datetime", "time_y": "datetime"}
    col_src = "time_x"
    col_tar = "time_y"
    result = mi_sim_instance.calculate(test_data_time[col_src], test_data_time[col_tar], metadata)
    result1 = mi_sim_instance.calculate(test_data_time[col_src], test_data_time[col_src], metadata)
    result2 = mi_sim_instance.calculate(test_data_time[col_tar], test_data_time[col_src], metadata)

    assert result >= 0
    assert result <= 1
    assert np.isclose(result1, 1, atol=1e-9)
    assert np.isclose(result, result2, atol=1e-9)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
