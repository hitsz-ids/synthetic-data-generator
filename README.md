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

# 🚀 Synthetic Data Generator

<p style="font-size: small;">Switch Language:
    <a href="https://github.com/hitsz-ids/synthetic-data-generator/blob/main/README_ZH_CN.md" target="_blank">简体中文</a>
  </p>

</p>
</div>

Synthetic Data Generator (SDG) is a framework focused on quickly generating high-quality structured tabular data. It supports many single-table and multi-table data synthesis algorithms, achieving up to 120 times performance improvement, and supports differential privacy and other methods to enhance the security of synthesized data.

Synthetic data is generated by machines based on real data and algorithms, it does not contain sensitive information, but can retain the characteristics of real data.
There is no correspondence between synthetic data and real data, and it is not subject to privacy regulations such as GDPR and ADPPA.
In practical applications, there is no need to worry about the risk of privacy leakage.
High-quality synthetic data can also be used in various fields such as data opening, model training and debugging, system development and testing, etc.

## 🎉 Features

- high performance
  - Supports a wide range of statistical data synthesis algorithms to achieve up to 120x performance improvement, without the need for GPU devices;
  - Optimised for big data scenarios, effectively reducing memory consumption;
  - Continuously tracking the latest advances in academia and industry, and introducing support for excellent algorithms and models in a timely manner.
  - Provide distributed training support for deep learning models with frameworks such as torch.
- Privacy enhancements:
  - SDG supports differential privacy, anonymization and other methods to enhance the security of synthetic data.
- Easy to extend
  - Supports expansion of models, data processing, data connectors, etc. in the form of plug-in packages

Read [the latest API docs](https://synthetic-data-generator.readthedocs.io/en/latest/) for more details.

## 🔛 Quick Start

### Pre-build image

You can use pre-built images to quickly experience the latest features.

```bash
docker pull idsteam/sdgx:latest
```

### Local Install (Recommended)

At present, the code of this project is updated very quickly. We recommend that you use SDG by installing it through the source code.

```bash
git clone git@github.com:hitsz-ids/synthetic-data-generator.git
pip install .
# Or install from git
pip install git+https://github.com/hitsz-ids/synthetic-data-generator.git
```

### Install from PyPi

```bash
pip install sdgx
```

### Quick Demo of Single Table Data Generation and Metric

#### Demo code

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

#### Comparison

Real data are as follows：

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

Synthetic data are as follows：

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

## 🤝 Join Community

The SDG project was initiated by **Institute of Data Security, Harbin Institute of Technology**. If you are interested in out project, welcome to join our community. We welcome organizations, teams, and individuals who share our commitment to data protection and security through open source:

- Read [CONTRIBUTING](./CONTRIBUTING.md) before draft a pull request.
- Submit an issue by viewing [View First Good Issue](https://github.com/hitsz-ids/synthetic-data-generator/issues/new) or submit a Pull Request.

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->

<!-- prettier-ignore-start -->

<!-- markdownlint-disable -->

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://wh1isper.github.io/"><img src="https://avatars.githubusercontent.com/u/43375501?v=4?s=100" width="100px;" alt="Zhongsheng Ji"/><br /><sub><b>Zhongsheng Ji</b></sub></a><br /><a href="#code-Wh1isper" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MooooCat"><img src="https://avatars.githubusercontent.com/u/141886018?v=4?s=100" width="100px;" alt="MoooCat"/><br /><sub><b>MoooCat</b></sub></a><br /><a href="#code-MooooCat" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joeyscave"><img src="https://avatars.githubusercontent.com/u/72662648?v=4?s=100" width="100px;" alt="YUAN KAIWEN"/><br /><sub><b>YUAN KAIWEN</b></sub></a><br /><a href="#code-joeyscave" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/sjh120"><img src="https://avatars.githubusercontent.com/u/86507761?v=4?s=100" width="100px;" alt="sjh120"/><br /><sub><b>sjh120</b></sub></a><br /><a href="#code-sjh120" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Z712023"><img src="https://avatars.githubusercontent.com/u/132286135?v=4?s=100" width="100px;" alt="Z712023"/><br /><sub><b>Z712023</b></sub></a><br /><a href="#code-Z712023" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://femilawal.com"><img src="https://avatars.githubusercontent.com/u/33192240?v=4?s=100" width="100px;" alt="Oluwafemi Lawal"/><br /><sub><b>Oluwafemi Lawal</b></sub></a><br /><a href="#code-Femi-lawal" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/iokk3732"><img src="https://avatars.githubusercontent.com/u/141700052?v=4?s=100" width="100px;" alt="iokk3732"/><br /><sub><b>iokk3732</b></sub></a><br /><a href="#code-iokk3732" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->

<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## 👩‍🎓 Related Work

### Research Paper

- CTGAN：[Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)
- TVAE：[Modeling Tabular Data using Conditional GAN](https://proceedings.neurips.cc/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html)
- table-GAN：[Data Synthesis based on Generative Adversarial Networks](https://arxiv.org/pdf/1806.03384.pdf)
- CTAB-GAN:[CTAB-GAN: Effective Table Data Synthesizing](https://proceedings.mlr.press/v157/zhao21a/zhao21a.pdf)
- OCT-GAN: [OCT-GAN: Neural ODE-based Conditional Tabular GANs](https://arxiv.org/pdf/2105.14969.pdf)

### Dataset

- [Adult](http://archive.ics.uci.edu/ml/datasets/adult)
- [Satellite](http://archive.ics.uci.edu/dataset/146/statlog+landsat+satellite)
- [Rossmann](https://www.kaggle.com/competitions/rossmann-store-sales/data)
- [Telstra](https://www.kaggle.com/competitions/telstra-recruiting-network/data)

## 📄 License

The SDG open source project uses Apache-2.0 license, please refer to the [LICENSE](https://github.com/hitsz-ids/synthetic-data-generator/blob/main/LICENSE).
