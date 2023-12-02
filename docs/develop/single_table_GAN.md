# SDG 神经网络模型开发文档

## 为 SDG 开发可运行的单表模型模块

本文档描述了如何开发模型算法模块，使得该模块可以在 sdgx 架构下被调用。

要开发模块，需要执行以下 5 个步骤：

1. 明确需要开发的模块类型；
1. 算法模块需要继承`BaseSynthesizerModel`类，并完成几个指定的函数；
1. 定义模型所需的`Discriminator`类（可选）；
1. 定义模型所需的`Generator`类（可选）；
1. 本地安装以及测试您的模型。

在以下各节中，我们将通过 `CTGAN` 这一例子详细描述这 5 个步骤。

## 第一步：明确需要开发的模块类型

首先，您需要明确需要开发的模型类型，目前我们支持：

- 单表模型（基于GAN的）
- 单表模型（基于统计学方法的）
- 多表模型
- 序列模型
- 其他

目前，其他模块仍在开发中，我们目前给出了单表模块（基于GAN）的开发文档。

如果您正在开发的基于GAN的单表模块，请继续往下看。

## 第二步：定义您的模型类

大体上讲，定义一个算法模块需要继承`BaseSynthesizerModel`基类，并完成几个指定的函数，即可成为您自己实现的模型模块。

其具体步骤如下：

1. 在 [single_tablem目录](../../sdgx/models/single_table/) 中创建名为 xxx.py 的 Python 脚本文件，其中 xxx 是您打算开发的模块。

1. 继承 `BaseSynthesizerModel`基类 。

   - 首先从 `sdgx/models/base.py` 中导入基类，并且导入其他必要的 Python 包，例如：

     ```python
         import warnings
         import numpy as np
         import pandas as pd
         import torch
         from torch import optim
         from torch.nn import (
             BatchNorm1d,
             Dropout,
             LeakyReLU,
             Linear,
             Module,
             ReLU,
             Sequential,
             functional,
         )
         from sdgx.models.base import BaseSynthesizerModel
         from sdgx.transform.sampler import DataSamplerCTGAN
         from sdgx.transform.transformer import DataTransformerCTGAN
     ```

   - 完成您的模块中的 `__init__` 函数，并定义相应的类变量，以CTGAN为例：

     ```python
       class CTGAN(BaseSynthesizerModel):
           def __init__(
               self,
               embedding_dim=128,
               generator_dim=(256, 256),
               discriminator_dim=(256, 256),
               generator_lr=2e-4,
               generator_decay=1e-6
               # ...
               # 本文档仅为示意，篇幅原因，更多参数已省略
           ):
               assert batch_size % 2 == 0

               self._embedding_dim = embedding_dim
               self._generator_dim = generator_dim
               self._discriminator_dim = discriminator_dim

               self._generator_lr = generator_lr
               self._generator_decay = generator_decay
               self._discriminator_lr = discriminator_lr
               self._discriminator_decay = discriminator_decay

               # ...
               # 本文档仅为示意，篇幅原因，更多参数已省略
     ```

   - 为了顺利使用sdg，您必须完成 `fit` 与 `sample` 这两个方法，它们有关仿真数据模型训练与数据的生成。

## 第三步：定义模型所需的`Discriminator`类（可选）

仅在此模块需要使用生成对抗网络（GAN）技术时候才需要执行此步骤。

您需要在同一个目录中定义`Discriminator`类，并实现 `__init__`、```` calc_gradient_penalty``` 以及  ````forward\`方法。

以CTGAN为例：

```python
class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()

    def calc_gradient_penalty(self, real_data, fake_data, device="cpu", pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        # 部分代码被省略

        return gradient_penalty

    def forward(self, input_):
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))
```

## 第四步：定义模型所需的`Generator`类（可选）

仅在此模块需要使用生成对抗网络（GAN）技术时候才需要执行此步骤。

您需要在同一个目录中定义`Generator`类，并实现 `__init__`以及 `forward`方法。

以CTGAN为例：

```python
class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        # 处于文档说明目的，部分代码被省略
        self.seq = Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)
        return data
```

## 第五步: 本地安装以及测试您的模型

在完成模块的代码编写后，您可以通过 `example/` 目录中的示例代码，将其中模型替换为您开发的模型，初步测试模型是否可用。

单元测试模块将在后续工作中逐步补充。

目前 Log 模块还尚未完成开发，未来，您还可以在以下路径中检查日志：`$PROJECT_DIR/log/{your jobid}`.
