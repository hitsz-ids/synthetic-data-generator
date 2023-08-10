# 运行该例子，可使用：
# ipython -i  example/1_ctgan_example.py 
# 并查看 sampled_data 变量

from utils.io.csv_utils import *
from models.single_table.ctgan import GeneratorCTGAN
from transform.transformer import DataTransformerCTGAN
from transform.sampler import DataSamplerCTGAN

# 针对 csv 格式的小规模数据
# 目前我们以 df 作为输入的数据的格式
demo_data, discrete_cols  = get_demo_single_table()


model = GeneratorCTGAN(epochs=10,\
                       transformer= DataTransformerCTGAN,\
                       sampler=DataSamplerCTGAN)
model.fit(demo_data, discrete_cols)

# sampled
sampled_data = model.generate(1000)


