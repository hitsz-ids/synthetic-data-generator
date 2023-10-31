# 本例用于测试 transformer 对象的正常使用

import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.transform.transformer_opt import DataTransformer
from sdgx.utils.io.csv_utils import *


def test_transformer_original():
    demo_data, discrete_cols = get_demo_single_table()
    ctgan_transformer = DataTransformerCTGAN()
    ctgan_transformer.fit(demo_data, discrete_cols)
    transformed_data = ctgan_transformer.transform(demo_data)


def test_transformer_opt():
    # 测试经过内存优化之后的 transformer
    demo_data_path = "./dataset/adult.csv"
    _, discrete_cols = get_demo_single_table()
    opt_transformer = DataTransformer()
    opt_transformer.fit(demo_data_path, discrete_cols)
    opt_transformer.transform(demo_data_path, "./output_tmp.csv")
    # inverse transforme
    x = pd.read_csv("./output_tmp.csv")
    x = x.to_numpy()
    inverse_x = opt_transformer.inverse_transform(x)
    inverse_x = opt_transformer.inverse_transform(
        x, output_path="inverse_tmp.csv", write_header=True
    )

    # remove tmp files
    shutil.os.remove("inverse_tmp.csv")
    shutil.os.remove("output_tmp.csv")
    pass


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
