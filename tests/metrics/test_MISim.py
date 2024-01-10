import numpy as np
import pandas as pd
import pytest

from sdgx.metrics.multi_table.mutual_information import MISim

# 创建测试数据
real_data_discrete = pd.DataFrame(
    {
        "col1": ["a", "b", "c", "d", "e"],
        "col2": ["a", "b", "b", "b", "e"],
    }
)

synthetic_data_discrete = pd.DataFrame(
    {
        "col1": ["a", "c", "d", "b", "b"],
        "col2": ["a", "c", "a", "a", "e"],
    }
)

real_data_cotinuous = pd.DataFrame(
    {
        "col1": [1, 1, 2, 2, 3],
        "col2": [4, 4, 5, 5, 6],
    }
)

synthetic_data_cotinuous = pd.DataFrame(
    {
        "col1": [1, 2, 2, 3, 3],
        "col2": [4, 5, 5, 6, 6],
    }
)

# 创建 JSD 实例
mi_sim = JSD()


def test_MISim_discrete():
    cols = ["col1", "col2"]
    result = mi_sim.calculate(real_data_discrete, synthetic_data_discrete, discrete=True)
    result1 = mi_sim.calculate(real_data_discrete, real_data_discrete, discrete=True)
    result2 = mi_sim.calculate(synthetic_data_discrete, real_data_discrete, discrete=True)

    assert result >= 0
    assert result <= 1
    assert result1 == 1
    assert result2 == result


def test_MISim_continuous():
    cols = ["col1"]
    result = mi_sim.calculate(real_data_cotinuous, synthetic_data_cotinuous, discrete=False)
    result1 = mi_sim.calculate(real_data_cotinuous, real_data_cotinuous, discrete=False)
    result2 = mi_sim.calculate(synthetic_data_cotinuous, real_data_cotinuous, discrete=False)

    assert result >= 0
    assert result <= 1
    assert result1 == 1
    assert result2 == result
