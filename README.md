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

### Quick Demo of Single Table Data Generation

```python
# Import modules
from sdgx.models.single_table.ctgan import CTGAN
from sdgx.utils.io.csv_utils import *

# Read data from demo
demo_data, discrete_cols  = get_demo_single_table()
```

Real data are as follows：

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
# Define model
model = CTGAN(epochs=10)
# Model training
model.fit(demo_data, discrete_cols)

# Generate synthetic data
sampled_data = model.generate(1000)
```

Synthetic data are as follows：

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
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MooooCat"><img src="https://avatars.githubusercontent.com/u/141886018?v=4?s=100" width="100px;" alt="MoooCat"/><br /><sub><b>MoooCat</b></sub></a><br /><a href="#code-MooooCat" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://wh1isper.github.io/"><img src="https://avatars.githubusercontent.com/u/43375501?v=4?s=100" width="100px;" alt="Zhongsheng Ji"/><br /><sub><b>Zhongsheng Ji</b></sub></a><br /><a href="#code-Wh1isper" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joeyscave"><img src="https://avatars.githubusercontent.com/u/72662648?v=4?s=100" width="100px;" alt="YUAN KAIWEN"/><br /><sub><b>YUAN KAIWEN</b></sub></a><br /><a href="#code-joeyscave" title="Code">💻</a></td>
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
