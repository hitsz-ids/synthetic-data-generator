# Refer CTGAN Version 0.6.0: https://github.com/sdv-dev/CTGAN@a40570e321cb46d798a823f350e1010a0270d804
# Which is Lincensed by MIT License

import warnings
from typing import Any, List, Optional

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

from sdgx.models.base import SynthesizerModel
from sdgx.models.components.sample import DataSamplerCTGAN
from sdgx.models.components.transform import DataTransformer
from sdgx.models.components.utils import random_state
from sdgx.models.extension import hookimpl
from sdgx.models.ml.single_table.base import TorchSynthesizerModel


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device="cpu", pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


# 目前针对了较大的数据进行内存占用上的优化
# 总体思路是采用分批 load 进内存的方法，优化其占用空间的大小
class CTGAN(TorchSynthesizerModel):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        memory_optimize(bool):
            设计上这个参数需要是 str 类型或者 bool 类型 可以是如下数值：
                - True：根据实际情况进行自动化内存优化，保证在能跑完的情况下，尽量多地利用内存，同时不报错
                - False：不采用任何内存优化策略
                - None：同 False
            该参数默认设为 False
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        memory_optimize=False,
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)

        self._generator = None

        # 是否启用内存优化，这里初步打算采用内存限制进行
        # 设计上这个参数需要是 str 类型或者 bool 类型 可以是如下数值：
        #   - True：根据实际情况进行自动化内存优化，保证在能跑完的情况下，尽量多地利用内存，同时不报错
        #   - False：不采用任何内存优化策略
        self.memory_optimize = memory_optimize

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """
        For compatibility, gumbelsoftmax is stable now: https://github.com/pytorch/pytorch/issues/41663
        """
        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == "tanh":
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == "softmax":
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f"Unexpected activation function {span_info.activation_fn}.")

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError("``train_data`` should be either pd.DataFrame or np.array.")

        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    # OPTIMIZE 新增方法，完成在内存受限情况下的 CTGAN 训练
    @random_state
    def fit_optimze(self, train_data_iterator, discrete_columns: Optional[List] = [], epoches=10):
        """OPTIMIZE 新增方法，在内存受限情况下完成 CTGAN 训练

        参数列表：
            train_data_iterator (iterator)：
                train_data_iterator 是 带有训练数据的迭代器，
                需要根据 csv 文件，迭代返回训练数据，
                属于必填参数。
            discrete_columns (list-like)：
                描述离散特征的一个 python 列表。
                如果 训练数据的迭代器 中不含有列名，则返回列的编号，
                否则 ，返回列的名称（可以乱序），
                属于选填参数，默认为 python 空列表。
            epoches(int-like)：
                CTGAN 模型的迭代次数，
                选填参数，默认为10，但严肃应用中应必须填写。
        """
        if epochs is None:
            epochs = self._epochs

        pass

    @random_state
    def fit(
        self,
        train_data,
        discrete_columns: Optional[List] = None,
        train_data_iterator=None,
        epochs=None,
    ):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            train_data_iterator (iterator)：
                train_data_iterator 是 带有训练数据的迭代器
                需要根据 csv 文件，迭代返回训练数据
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        # set discrete_columns
        if not discrete_columns:
            discrete_columns = []

        # OPTIMIZE 检查 optimize 以及 train_data_iterator
        if self.memory_optimize and train_data_iterator is None:
            raise ValueError("train_data_iterator should not be None.")
        # 如果符合 optimize 的需求，则转到新实现的 optimize 方法，这样也不干扰老方法的顺利执行
        if self.memory_optimize and train_data_iterator is not None:
            self.fit_optimze(train_data_iterator, discrete_columns, epochs)
            return
        # OPTIMIZE 改动结束

        # 以下为原始的 fit 方法
        self._validate_discrete_columns(train_data, discrete_columns)

        # 参数检查
        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        # 载入 transformer
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        # 使用 transformer 处理数据
        train_data = self._transformer.transform(train_data)

        # 载入 sampler
        self._data_sampler = DataSamplerCTGAN(
            train_data, self._transformer.output_info_list, self._log_frequency
        )

        # data dim 从 transformer 中取得
        data_dim = self._transformer.output_dimensions

        # sampler 作为参数给到 Generator 以及 Discriminator
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        # 初始化 optimizer G 以及 D
        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        # 开始执行 fit  流程，直到结束
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm]
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            if self._verbose:
                print(
                    f"Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},"  # noqa: T001
                    f"Loss D: {loss_d.detach().cpu(): .4f}",
                    flush=True,
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)


@hookimpl
def register(manager):
    manager.register("CTGAN", CTGAN)
