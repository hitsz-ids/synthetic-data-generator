# 本例用于测试 transformer 对象的正常使用

import os
import pytest

_HERE = os.path.dirname(__file__)

import sys 
sys.path.append(os.getcwd())

from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.utils.io.csv_utils import *

def test_transformer_original():
    demo_data, discrete_cols = get_demo_single_table()
    ctgan_transformer = DataTransformerCTGAN()
    ctgan_transformer.fit(demo_data, discrete_cols)
    transformed_data = ctgan_transformer.transform(demo_data)

def test_transformer():
    # 测试经过内存优化之后的 transformer 
    demo_data, discrete_cols = get_demo_single_table()
    
    pass

if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
