# 运行该例子，可使用：
# ipython -i  example/2_guassian_copula_example.py
# 并查看 sampled_data 变量

from sdgx.statistics.single_table.copula import GaussianCopulaSynthesizer
from sdgx.utils.io.csv_utils import *

# 针对 csv 格式的小规模数据
# 目前我们以 df 作为输入的数据的格式
demo_data, discrete_cols = get_demo_single_table()
# print(demo_data)
# print(discrete_cols)

model = GaussianCopulaSynthesizer(discrete_cols)
model.fit(demo_data)

# sampled
sampled_data = model.sample(10)
print(sampled_data)
