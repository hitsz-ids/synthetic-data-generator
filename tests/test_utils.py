# 本例用于测试 transformer 对象的正常使用

import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.append(os.getcwd())

from sdgx.utils.io.csv_utils import *


def test_csv_tools():
    pass


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
