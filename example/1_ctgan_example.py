# 运行该例子，可使用：
# ipython -i  example/1_ctgan_example.py
# 并查看 sampled_data 变量
import numpy as np

from sdgx.data_process.transform.transform import DataTransformer
from sdgx.models.single_table.ctgan import CTGAN

# from sdgx.data_process.sampling.sampler import DataSamplerCTGAN
# from sdgx.data_process.transform.transform import DataTransformer
from sdgx.utils.io.csv_utils import *
from sdgx.metrics.column.jsd import jsd
from scipy.stats import gaussian_kde

# 针对 csv 格式的小规模数据
# 目前我们以 df 作为输入的数据的格式
demo_data, discrete_cols = get_demo_single_table()
jsD = jsd()

# model = CTGAN(epochs=10)
# model.fit(demo_data, discrete_cols)

# sampled_data = model.sample(1000)
output_file_path = Path("./dataset/test.csv")  # 修改为你想要保存的路径和文件名
sampled_data = pd.read_csv(output_file_path)




# Extract the data from the DataFrame
selected_columns = ['fnlwgt']
metrics = jsD.calculate(demo_data,sampled_data,False,selected_columns)
print(metrics)

