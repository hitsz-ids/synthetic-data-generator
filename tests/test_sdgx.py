import os

import pytest

_HERE = os.path.dirname(__file__)

import sys

sys.path.append(os.getcwd())


from sdgx.models.single_table.ctgan import CTGAN
from sdgx.utils.io.csv_utils import *


def test_helloworld():
    # 读取数据
    demo_data, discrete_cols = get_demo_single_table()
    model = CTGAN(epochs=1)
    model.fit(demo_data, discrete_cols)

    # 生成合成数据
    sampled_data = model.sample(1000)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
