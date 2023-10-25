import os

import pytest

from sdgx.statistics.single_table.ctabgan import CTABGAN

_HERE = os.path.dirname(__file__)

from sdgx.transform.sampler import DataSamplerCTGAN
from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.utils.io.csv_utils import *


def test_ctabgan():
    # 读取数据
    demo_data, discrete_cols = get_demo_single_table()
    model = CTABGAN(epochs=1, transformer=DataTransformerCTGAN, sampler=DataSamplerCTGAN)
    model.fit(demo_data, discrete_cols)

    # 生成合成数据
    sampled_data = model.sample(1000)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
