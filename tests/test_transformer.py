# 本例用于测试 transformer 对象的正常使用

import os
import pytest

_HERE = os.path.dirname(__file__)

import sys 
sys.path.append(os.getcwd())

from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.utils.io.csv_utils import *
from sdgx.transform.transformer_opt import DataTransformer

def test_transformer_original():
    demo_data, discrete_cols = get_demo_single_table()
    ctgan_transformer = DataTransformerCTGAN()
    ctgan_transformer.fit(demo_data, discrete_cols)
    transformed_data = ctgan_transformer.transform(demo_data)

def test_transformer_opt():
    # 测试经过内存优化之后的 transformer 
    demo_data, discrete_cols = get_demo_single_table()
    demo_data_path = "./dataset/adult.csv"
    opt_transformer = DataTransformer()
    opt_transformer.fit(demo_data_path, discrete_cols)
    output_path = "tmp.csv"
    opt_data_path = opt_transformer.transform(demo_data_path)
    
    pass

if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
