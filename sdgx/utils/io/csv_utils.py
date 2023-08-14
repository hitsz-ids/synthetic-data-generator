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


# 自动检测函数，用于检测 discrete_cols
# 不一定最准确，但可以一定程度上方便使用
def auto_select_discrete_cols(pd_obj):
    pass
