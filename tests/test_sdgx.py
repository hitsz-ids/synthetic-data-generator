import os

import pytest

_HERE = os.path.dirname(__file__)


from sdgx.models.single_table.ctgan import GeneratorCTGAN
from sdgx.transform.sampler import DataSamplerCTGAN
from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.utils.io.csv_utils import *


def test_helloworld():
    # 读取数据
    demo_data, discrete_cols = get_demo_single_table()
    model = GeneratorCTGAN(epochs=1, transformer=DataTransformerCTGAN, sampler=DataSamplerCTGAN)
    # 训练模型
    model.fit(demo_data, discrete_cols)

    # 生成合成数据
    sampled_data = model.generate(1000)


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
