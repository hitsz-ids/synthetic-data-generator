import pandas as pd 


def get_demo_single_table():
    demo_data_path = "dataset/adult.csv"
    pd_obj = pd.read_csv(demo_data_path)
    discrete_cols = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]
    return pd_obj, discrete_cols 


# 获取csv形式的单表数据
def get_single_table(input_path):

    pass

# 自动检测函数，用于检测 discrete_cols
# 不一定最准确，但可以一定程度上方便使用
def auto_select_discrete_cols(pd_obj):

    pass
