import numpy as np
import pandas as pd
import pytest
from sdgx.metrics.column.jsd import JSD

# 创建测试数据
real_data_discrete = pd.DataFrame({
    'col1': ['a', 'b', 'c', 'd', 'e'],
    'col2': ['a', 'b', 'b', 'b', 'e'],
})

synthetic_data_discrete = pd.DataFrame({
    'col1': ['a', 'c', 'd', 'b', 'b'],
    'col2': ['a', 'c', 'a', 'a', 'e'],
})

real_data_cotinuous = pd.DataFrame({
    'col1': [1, 1, 2, 2, 3],
    'col2': [4, 4, 5, 5, 6],
})

synthetic_data_cotinuous = pd.DataFrame({
    'col1': [1, 2, 2, 3, 3],
    'col2': [4, 5, 5, 6, 6],
})

# 创建 JSD 实例
jsd_instance = JSD()


def test_jsd_discrete():
    cols = ['col1', 'col2']
    result = jsd_instance.calculate(real_data_discrete, synthetic_data_discrete, cols, discrete=True)
    result1 = jsd_instance.calculate(real_data_discrete, real_data_discrete, cols, discrete=True)
    result2 = jsd_instance.calculate(synthetic_data_discrete, real_data_discrete, cols, discrete=True)

    assert result >= 0
    assert result <= 1
    assert result1 == 0
    assert result2 == result



def test_jsd_continuous():
    cols = ['col1']
    result = jsd_instance.calculate(real_data_cotinuous, synthetic_data_cotinuous, cols, discrete=False)
    result1 = jsd_instance.calculate(real_data_cotinuous, real_data_cotinuous, cols, discrete=False)
    result2 = jsd_instance.calculate(synthetic_data_cotinuous, real_data_cotinuous, cols, discrete=False)

    assert result >= 0
    assert result <= 1
    assert result1 == 0
    assert result2 == result