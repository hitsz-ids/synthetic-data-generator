from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from sdgx.metrics.column.jsd import JSD


# 创建测试数据
@pytest.fixture
def dummy_data(dummy_single_table_path):
    yield pd.read_csv(dummy_single_table_path)


@pytest.fixture
def test_data():
    role_set = ["admin", "user", "guest"]
    df = pd.DataFrame(
        {
            "role": [random.choice(role_set) for _ in range(10)],
            "feature_x": [random.random() for _ in range(10)],
        }
    )
    return df


@pytest.fixture
def jsd_instance():
    return JSD()


def test_jsd_discrete(dummy_data, test_data, jsd_instance):
    cols = ["role"]
    result = jsd_instance.calculate(dummy_data, test_data, cols, discrete=True)
    result1 = jsd_instance.calculate(dummy_data, dummy_data, cols, discrete=True)
    result2 = jsd_instance.calculate(test_data, dummy_data, cols, discrete=True)

    assert result >= 0
    assert result <= 1
    assert np.isclose(result1, 0, atol=1e-9)
    assert np.isclose(result, result2, atol=1e-9)


def test_jsd_continuous(dummy_data, test_data, jsd_instance):
    cols = ["feature_x"]
    result = jsd_instance.calculate(dummy_data, test_data, cols, discrete=False)
    result1 = jsd_instance.calculate(dummy_data, dummy_data, cols, discrete=False)

    assert result >= 0
    assert result <= 1
    assert np.isclose(result1, 0, atol=1e-9)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
