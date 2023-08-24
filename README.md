# 合成数据生成器 -- 快速生成高质量合成数据！

合成数据生成器（Synthetic Data Generator，SDG）是一个专注于快速生成高质量结构化表格数据的组件。支持10余种单表、多表数据合成算法，实现最高120倍性能提升，支持差分隐私等方法，加强合成数据安全性。

合成数据是由机器根据真实数据与算法生成的，合成数据不含敏感信息，但能保留真实数据中的行为特征。合成数据与真实数据不存在任何对应关系，不受 GDPR 、ADPPA等隐私法规的约束，在实际应用中不需要担心隐私泄漏风险。高质量的合成数据可用于数据安全开放、模型训练调试、系统开发测试等众多领域。

| 重要链接                                                                                                                                                                                                   |                                                       |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 📖  [文档](https://github.com/hitsz-ids/synthetic-data-generator/tree/main/docs)                                                                                                                                                                            | 项目API文档                                           |
| :octocat:  [项目仓库](https://github.com/hitsz-ids/synthetic-data-generator) | 项目Github仓库                                        |
| 📜 [License](https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE)                                                                                                                         | Apache-2.0 license                                    |
| 举个例子 🌰                                                                                                                                                                                                | 在[AI靶场](https://datai.pcl.ac.cn/)上运行SDG示例（TBD） |

## 目录


- [快速开始](#快速开始)
- [主要特性](#主要特性)
- [算法列表](#算法列表)
- [相关论文和数据集链接](#相关论文和数据集链接)
- [API](#API)
- [维护者](#维护者)
- [如何贡献](#如何贡献)
- [许可证](#许可证)

## 快速开始

### 从Pypi安装

```bash
pip install sdgx
```

### 单表数据快速合成示例

```python
# 导入相关模块
from sdgx.utils.io.csv_utils import *
from sdgx.models.single_table.ctgan import GeneratorCTGAN
from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.transform.sampler import DataSamplerCTGAN

# 读取数据
demo_data, discrete_cols  = get_demo_single_table()
```

真实数据示例如下：

```
       age  workclass  fnlwgt  ... hours-per-week  native-country  class
0       27    Private  177119  ...             44   United-States  <=50K
1       27    Private  216481  ...             40   United-States  <=50K
2       25    Private  256263  ...             40   United-States  <=50K
3       46    Private  147640  ...             40   United-States  <=50K
4       45    Private  172822  ...             76   United-States   >50K
...    ...        ...     ...  ...            ...             ...    ...
32556   43  Local-gov   33331  ...             40   United-States   >50K
32557   44    Private   98466  ...             35   United-States  <=50K
32558   23    Private   45317  ...             40   United-States  <=50K
32559   45  Local-gov  215862  ...             45   United-States   >50K
32560   25    Private  186925  ...             48   United-States  <=50K

[32561 rows x 15 columns]

```

```python
#定义模型
model = GeneratorCTGAN(epochs=10,\
                       transformer= DataTransformerCTGAN,\
                       sampler=DataSamplerCTGAN)
# 训练模型
model.fit(demo_data, discrete_cols)

# 生成合成数据
sampled_data = model.generate(1000)
```

合成数据如下：

```
   age         workclass  fnlwgt  ... hours-per-week  native-country  class
0   33           Private  276389  ...             41   United-States   >50K
1   33  Self-emp-not-inc  296948  ...             54   United-States  <=50K
2   67       Without-pay  266913  ...             51        Columbia  <=50K
3   49           Private  423018  ...             41   United-States   >50K
4   22           Private  295325  ...             39   United-States   >50K
5   63           Private  234140  ...             65   United-States  <=50K
6   42           Private  243623  ...             52   United-States  <=50K
7   75           Private  247679  ...             41   United-States  <=50K
8   79           Private  332237  ...             41   United-States   >50K
9   28         State-gov  837932  ...             99   United-States  <=50K
```

## 主要特性

+ 高性能
  + 支持10余种单表、多表数据合成算法，实现最高120倍性能提升；
  + SDG会持续跟踪学术界和工业界的最新进展，及时引入支持优秀算法和模型。
+ 生产环境快速部署
  + 针对实际生产需求进行优化，提升模型性能，降低内存开销，支持单机多卡、多机多卡等实用特性；
  + 提供自动化部署、容器化技术、自动化监控和报警等生产环境所需技术，支持容器化快速一键部署；
  + 针对负载均衡和容错性进行专门优化，提升组件可用性。
+ 隐私增强：
  + 提供中文敏感数据自动识别能力，包括姓名、身份证号、人名等17种常见敏感字段；
  + 支持差分隐私、匿名化等方法，加强合成数据安全性。

## 算法列表

### 表1：单表合成算法效果对比(F1-score)

|    模型    | Adult(二分类数据集)(%) | Satellite(多分类数据集)(%) |
| :--------: | :--------------------: | :------------------------: |
| 原始数据集 |          69.5          |           89.23           |
|   CTGAN   |         60.38         |           69.43           |
|    TVAE    |         59.52         |           83.58           |
| table-GAN |         63.29         |           79.15           |
|  CTAB-GAN  |         58.59         |           79.24           |
|  OCT-GAN  |         55.18         |           80.98           |
|  CorTGAN  |    **67.13**    |      **84.27**      |

### 表2：多表合成算法效果对比

|    模型    | Rossmann(回归数据集)(rmspe) | Telstra(分类数据集)(mlogloss) |
| :--------: | :-------------------------: | :---------------------------: |
| 原始数据集 |           0.2217           |            0.5381            |
|    SDV    |           0.6897           |            1.1719            |
|   CWAMT   |      **0.4348**      |        **0.818**        |

### 相关论文和数据集链接

#### 论文

- CTGAN：[Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)
- TVAE：[Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)
- table-GAN：[Data Synthesis based on Generative Adversarial Networks](https://arxiv.org/pdf/1806.03384.pdf)
- CTAB-GAN:[CTAB-GAN: Effective Table Data Synthesizing](https://proceedings.mlr.press/v157/zhao21a/zhao21a.pdf)
- OCT-GAN: [OCT-GAN: Neural ODE-based Conditional Tabular GANs](https://arxiv.org/pdf/2105.14969.pdf)
- SDV：[The Synthetic data vault](https://sci-hub.se/10.1109/DSAA.2016.49 "多表合成")

#### 数据集

- [Adult数据集](http://archive.ics.uci.edu/ml/datasets/adult)
- [Satellite数据集](http://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)
- [Rossmann数据集](https://www.kaggle.com/competitions/rossmann-store-sales/data)
- [Telstra数据集](https://www.kaggle.com/competitions/telstra-recruiting-network/data)


## API

具体接口参数请参考 [API文档](https://SDG.readthedocs.io/en/latest/api/index.html) 【TBD】。

## 维护者

SDG开源项目由**哈尔滨工业大学（深圳）数据安全研究院**发起，若您对SDG项目感兴趣并愿意一起完善它，欢迎加入我们的开源社区。

## 如何贡献

非常欢迎你的加入！[提一个 Issue](https://github.com/hitsz-ids/synthetic-data-generator/issues/new) 或者提交一个 Pull Request。

开发环境配置请参考[开发者文档](./DEVELOP.md)

## 许可证

SDG开源项目使用 Apache-2.0 license，有关协议请参考[LICENSE](https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE)。

[文档]: https://sgd.github.io/
[项目仓库]: https://github.com/hitsz-ids/synthetic-data-generator
[License]: https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE
[AI靶场]: https://datai.pcl.ac.cn/
