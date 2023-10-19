import urllib.request
from pathlib import Path

import pandas as pd


def get_demo_single_table():
    demo_data_path = Path("dataset/adult.csv")
    if not demo_data_path.exists():
        # Download from datahub
        demo_data_path.parent.mkdir(parents=True, exist_ok=True)

        # FIXME: Use logging
        print("Downloading demo data from datahub.io to ./dataset/adult.csv")
        url = "https://datahub.io/machine-learning/adult/r/adult.csv"
        urllib.request.urlretrieve(url, demo_data_path)

    pd_obj = pd.read_csv(demo_data_path)
    discrete_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "class",
    ]
    return pd_obj, discrete_cols


# 获取csv形式的单表数据
def get_single_table(input_path):
    pass


def get_csv_column(input_path, column_name):
    '''按 column 形式读取 csv，返回 pd.DataFrame 的单列数据

    输入参数:
        input_path (str):
            作为输入的 csv 路径
        column_name (str)：
            需要提取的 csv 列名

    返回对象说明:
        namedtuple对象 (pd.DataFrame):
            返回单个 ``pd.DataFrame`` 对象
    '''
    df_col = pd.read_csv(input_path, usecols=[column_name])
    return df_col
