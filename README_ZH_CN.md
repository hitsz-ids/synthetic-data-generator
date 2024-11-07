<div align="center">
  <img src="assets/sdg_logo.png" width="400" >
</div>

<div align="center">
<p align="center">

<p align="center">
<a href="https://github.com/hitsz-ids/synthetic-data-generator/actions"><img alt="Actions Status" src="https://github.com/hitsz-ids/synthetic-data-generator/actions/workflows/python-package.yml/badge.svg"></a>
<a href='https://synthetic-data-generator.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/synthetic-data-generator/badge/?version=latest' alt='Documentation Status' /></a>
<a href="https://results.pre-commit.ci/latest/github/hitsz-ids/synthetic-data-generator/main"><img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/hitsz-ids/synthetic-data-generator/main.svg"></a>
<a href="https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/github/license/hitsz-ids/synthetic-data-generator"></a>
<a href="https://github.com/hitsz-ids/synthetic-data-generator/releases/"><img alt="Releases" src="https://img.shields.io/github/v/release/hitsz-ids/synthetic-data-generator"></a>
<a href="https://github.com/hitsz-ids/synthetic-data-generator/releases/"><img alt="Pre Releases" src="https://img.shields.io/github/v/release/hitsz-ids/synthetic-data-generator?include_prereleases&label=pre-release&logo=github"></a>
<a href="https://github.com/hitsz-ids/synthetic-data-generator"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/hitsz-ids/synthetic-data-generator"></a>
<a href="https://github.com/hitsz-ids/synthetic-data-generator"><img alt="Python version" src="https://img.shields.io/pypi/pyversions/sdgx"></a>
<a href="https://github.com/hitsz-ids/synthetic-data-generator/contributors"><img alt="contributors" src="https://img.shields.io/github/all-contributors/hitsz-ids/synthetic-data-generator?color=ee8449&style=flat-square"></a>
<a href="https://join.slack.com/t/hitsz-ids/shared_invite/zt-2395mt6x2-dwf0j_423QkAgGvlNA5E1g"><img alt="slack" src="https://img.shields.io/badge/slack-join%20chat-ff69b4.svg?style=flat-square"></a>
</p>

# 🚀 合成数据生成器 -- 快速生成高质量合成数据！

<p style="font-size: small;">切换语言:
    <a href="https://github.com/hitsz-ids/synthetic-data-generator/blob/main/README.md" target="_blank">English</a> &nbsp;| &nbsp;最新
    <a href="https://synthetic-data-generator.readthedocs.io/en/latest/" target="value">API文档</a>&nbsp;| &nbsp;
     <a href="ROADMAP_ZH_CN.md" target="value">项目发展规划</a> &nbsp;| &nbsp;
    加入 <a href="assets/live_QR_code.jpg" target="value">Wechat群组</a>
  </p>

<p style="font-size: small;">
    查看 Colab 例子:&nbsp;
    <a href="https://colab.research.google.com/drive/1VFnP59q3eoVtMJ1PvcYjmuXtx9N8C7o0?usp=sharing" target="value"> 使用LLM仿真数据</a>
    &nbsp;| &nbsp;
    <a href="https://colab.research.google.com/drive/1_chuTVZECpj5fklj-RAp7ZVrew8weLW_?usp=sharing" target="value"> 借助LLM进行表外特征推断</a>
    &nbsp;| &nbsp;
    <a href="https://colab.research.google.com/drive/1cMB336jN3kb-m_pr1aJjshnNep_6bhsf?usp=sharing" target="value">支持十亿级数据的CTGAN</a>
</p>

</p>
</div>

合成数据生成器（Synthetic Data Generator，SDG）是一个专注于快速生成高质量的结构化表格数据的数据组件。

合成数据（Synthetic Data）不包含任何敏感信息，但它保留了原始数据的基本特性，使其免于GDPR和ADPPA等隐私法规的约束，消除实际应用中的隐私泄露风险。

高质量的合成数据可以安全、多样化地在各种领域中使用，包括数据共享、模型训练和调试、系统开发和测试等应用。

我们很期待您的贡献，请通过这个[贡献概述指南](CONTRIBUTING_ZH_CN.md)开始项目！

## 💥 相关信息

我们的里程碑和时间节点如下所示：

🔥 2024年5月30日：Data Processor 模块被正式合并，该模块可以：1）可以帮助 SDG 将部分数据列（如 Datetime 类型的列）在送入模型前进行格式转换，从而避免被当作离散类型处理，对模型生成数据反向转换成原有格式；2）对各种数据类型进行更加定制化的预处理和后处理；3）轻松应对原始数据中的空值等问题；4）支持插件系统。

🔥 2024年2月20日：基于LLM的单表数据合成模型已包含，查看Colab示例：<a href="https://colab.research.google.com/drive/1VFnP59q3eoVtMJ1PvcYjmuXtx9N8C7o0?usp=sharing" target="value">LLM：数据合成</a> 和 <a href="https://colab.research.google.com/drive/1_chuTVZECpj5fklj-RAp7ZVrew8weLW_?usp=sharing" target="value">LLM：表外特征推断</a>。

🔧 2024年2月7日：SDG团队完善了 `sdgx.data_models.metadata`，支持描述单表、多表的元数据信息，支持多种数据类型，支持数据类型的自动推断能力，查看Colab示例：<a href="https://colab.research.google.com/drive/1b4ZTpgSYjOt7ekp1Wj8CxDknbOHEwA7s?usp=sharing" target="value">SDG Single-Table Metadata</a>。

🔶 2023年12月20日：v0.1.0发布，包含支持数十亿数据处理能力的CTGAN模型，查看我们的<a href="https://github.com/hitsz-ids/synthetic-data-generator/tree/ main/benchmarks#results" target="value"> 针对 SDV 的基准</a>，SDG 实现了更少的内存消耗并避免了训练期间的崩溃（Out of Memory），具体使用，请查看我们的Colab示例：<a href="https://colab.research.google.com/drive/1cMB336jN3kb-m_pr1aJjshnNep_6bhsf?usp=sharing" target="value">支持十亿级数据的CTGAN</a>。

🔆 2023年8月10日：第一行SDG代码提交。

## 🎉 借助LLM进行合成数据生成

长期以来，LLM一直被用来理解和生成各种类型的数据。 事实上，LLM在表格数据生成方面也有较强的性能。 且LLM还具有一些传统（基于GAN方法或统计方法）无法实现的能力。

我们的 `sdgx.models.LLM.single_table.gpt.SingleTableGPTModel` 实现了两个新功能：

### 无原始记录的数据合成功能

无需原始训练数据，可以根据元数据生成合成数据，查看 <a href="https://colab.research.google.com/drive/1VFnP59q3eoVtMJ1PvcYjmuXtx9N8C7o0?usp=sharing" target="value"> Colab 例子</a>。

![Synthetic data generation without Data](assets/LLM_Case_1.gif)

### 表外特征推断功能

根据表中已有的数据以及LLM掌握的知识推断表外特征，即新的列数据，查看 <a href="https://colab.research.google.com/drive/1_chuTVZECpj5fklj-RAp7ZVrew8weLW_?usp=sharing" target="value"> Colab 例子</a>。

![Off-Table feature inference](assets/LLM_Case_2.gif)

## 💫 Why SDG ?

- 无限进步：
  - 支持多种统计学数据合成算法，支持基于LLM的仿真数据生成方法；
  - 为大数据场景优化，有效减少内存消耗；
  - 持续跟踪学术界和工业界的最新进展，及时引入支持优秀算法和模型。
- 隐私增强：
  - 提供中文敏感数据自动识别能力，包括姓名、身份证号、人名等17种常见敏感字段；
  - 支持差分隐私、匿名化等方法，加强合成数据安全性。
- 易扩展：
  - 支持以插件包的形式拓展模型、数据处理、数据连接器等功能。

## 🌀 快速开始

### 预构建镜像

您可以使用预构建的镜像来快速体验最新功能。

```bash
docker pull idsteam/sdgx:latest
```

### 从Pypi安装

```bash
pip install sdgx
```

### 从本地安装

您可以通过源码进行安装的方式使用SDG。

```bash
git clone git@github.com:hitsz-ids/synthetic-data-generator.git
pip install .
# 或者直接从git安装
pip install git+https://github.com/hitsz-ids/synthetic-data-generator.git
```

### 单表数据快速合成示例

#### 演示代码

```python
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data

# This will download demo data to ./dataset
dataset_csv = download_demo_data()

# Create data connector for csv file
data_connector = CsvConnector(path=dataset_csv)

# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel(epochs=1),  # For quick demo
    data_connector=data_connector,
)

# Fit the model
synthesizer.fit()

# Sample
sampled_data = synthesizer.sample(1000)
print(sampled_data)
```

#### 对比

真实数据：

```python
>>> data_connector.read()
       age         workclass  fnlwgt  education  ...  capitalloss hoursperweek native-country  class
0        2         State-gov   77516  Bachelors  ...            0            2  United-States  <=50K
1        3  Self-emp-not-inc   83311  Bachelors  ...            0            0  United-States  <=50K
2        2           Private  215646    HS-grad  ...            0            2  United-States  <=50K
3        3           Private  234721       11th  ...            0            2  United-States  <=50K
4        1           Private  338409  Bachelors  ...            0            2           Cuba  <=50K
...    ...               ...     ...        ...  ...          ...          ...            ...    ...
48837    2           Private  215419  Bachelors  ...            0            2  United-States  <=50K
48838    4               NaN  321403    HS-grad  ...            0            2  United-States  <=50K
48839    2           Private  374983  Bachelors  ...            0            3  United-States  <=50K
48840    2           Private   83891  Bachelors  ...            0            2  United-States  <=50K
48841    1      Self-emp-inc  182148  Bachelors  ...            0            3  United-States   >50K

[48842 rows x 15 columns]

```

仿真数据：

```python
>>> sampled_data
     age workclass  fnlwgt     education  ...  capitalloss hoursperweek native-country  class
0      1       NaN   28219  Some-college  ...            0            2    Puerto-Rico  <=50K
1      2   Private  250166       HS-grad  ...            0            2  United-States   >50K
2      2   Private   50304       HS-grad  ...            0            2  United-States  <=50K
3      4   Private   89318     Bachelors  ...            0            2    Puerto-Rico   >50K
4      1   Private  172149     Bachelors  ...            0            3  United-States  <=50K
..   ...       ...     ...           ...  ...          ...          ...            ...    ...
995    2       NaN  208938     Bachelors  ...            0            1  United-States  <=50K
996    2   Private  166416     Bachelors  ...            2            2  United-States  <=50K
997    2       NaN  336022       HS-grad  ...            0            1  United-States  <=50K
998    3   Private  198051       Masters  ...            0            2  United-States   >50K
999    1       NaN   41973       HS-grad  ...            0            2  United-States  <=50K

[1000 rows x 15 columns]
```

## 👩‍🎓 相关工作

### 论文

- CTGAN：[Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)
- C3-TGAN: [C3-TGAN- Controllable Tabular Data Synthesis with Explicit Correlations and Property Constraints](https://www.researchgate.net/publication/374652636_C3-TGAN-_Controllable_Tabular_Data_Synthesis_with_Explicit_Correlations_and_Property_Constraints)
- TVAE：[Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)
- table-GAN：[Data Synthesis based on Generative Adversarial Networks](https://arxiv.org/pdf/1806.03384.pdf)
- CTAB-GAN:[CTAB-GAN: Effective Table Data Synthesizing](https://proceedings.mlr.press/v157/zhao21a/zhao21a.pdf)
- OCT-GAN: [OCT-GAN: Neural ODE-based Conditional Tabular GANs](https://arxiv.org/pdf/2105.14969.pdf)

## 🤝  如何贡献

SDG开源项目由**哈尔滨工业大学（深圳）数据安全研究院**发起，若您对SDG项目感兴趣并愿意一起完善它，欢迎加入我们的开源社区：

- 非常欢迎你的加入！[查看 Good First Issue](https://github.com/hitsz-ids/synthetic-data-generator/labels/good%20first%20issue) 或者提交一个 Pull Request。
- 开发环境配置请参考[开发者文档](./CONTRIBUTING.md)
- 加入微信群：

<div align="left">
  <img src="assets/live_QR_code.jpg" width="400" >
</div>

## 📄 许可证

SDG开源项目使用 Apache-2.0 license，有关协议请参考[LICENSE](https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE)。
