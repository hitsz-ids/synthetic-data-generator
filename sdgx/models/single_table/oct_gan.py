import warnings
from typing import List, Optional
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
from sdgx.utils.utils import *


class ODEBlockD(nn.Module):
    """ 常微分方程块 """
    def __init__(self, odefunc, num_split):
        """
            参数:
                odefunc:常微分方程函数
                num_split:切分积分时间
        """
        super(ODEBlockD, self).__init__()
        self.odefunc = odefunc
        self.num_split = num_split
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        """ x:初始值和积分时间 """
        initial_value = x[0]
        integration_time = torch.cat(x[1], dim = 0).to(self.device)
        zero = torch.tensor([0.], requires_grad=False).to(self.device)
        one = torch.tensor([1.], requires_grad=False).to(self.device)

        all_time = torch.cat( [zero, integration_time, one],dim=0).to(self.device)
        self.total_integration_time = [all_time[i:i+2] for i in range(self.num_split)]

        out = [[1, initial_value]]
        for i in range(len(self.total_integration_time)):
            self.integration_time = self.total_integration_time[i].type_as(initial_value)
            # 借助odeint求解常微分方程
            out_ode = odeint(self.odefunc, out[i][1], self.integration_time, rtol=1e-3, atol=1e-3)
            out.append(out_ode)

        return [i[1] for i in out]

class ODEFuncD(nn.Module):
    """ 常微分方程函数 """
    def __init__(self, first_layer_dim):
        super(ODEFuncD, self).__init__()
        self.layer_start = nn.Sequential(nn.BatchNorm1d(first_layer_dim),
                                    nn.ReLU())

        self.layer_t = nn.Sequential(nn.Linear(first_layer_dim + 1, first_layer_dim * 2),
                                     nn.BatchNorm1d(first_layer_dim * 2),
                                     nn.ReLU(),
                                     nn.Linear(first_layer_dim * 2, first_layer_dim * 1),
                                     nn.BatchNorm1d(first_layer_dim * 1),
                                     nn.ReLU())
        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out, tt],dim = 1)
        out = self.layer_t(out)
        return out

class Discriminator(Module):
    """ 判别器 """

    def __init__(self, input_dim, discriminator_dim, pac=10):
        """
            参数：
                input_dim:输入维度
                discriminator_dim:判别器维度
                pac:Packing,每次打包pac个样本，使模型更好地学习到真实样本分布和假样本分布之间的区别
        """
        super(Discriminator, self).__init__()
        self.num_split = 3
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        self.seq = Sequential(*seq)
        self.ode = ODEBlockD(ODEFuncD(dim), self.num_split)

        self.traj_dim = dim * (self.num_split + 1)
        self.last1 = nn.Linear(self.traj_dim, self.traj_dim * 2)
        self.last3 = nn.Linear(self.traj_dim * 2, self.traj_dim)
        self.last4 = nn.Linear(self.traj_dim, 1)

    def calc_gradient_penalty(self, real_data, fake_data, t_pairs, device='cpu', pac=1, lambda_=10):
        """
            计算梯度惩罚
            在真实数据和生成数据之间插值,计算插值点处的梯度
            用于限制判别器输出的范围，防止判别器过度拟合
        """
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self([interpolates, t_pairs])

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        value = input_[0]
        time = input_[1]
        out = self.seq(value.view(-1, self.pacdim))
        out1_time = [out, time]
        out = self.ode(out1_time)
        out = torch.cat(out, dim = 1)

        out = functional.leaky_relu(self.last1(out))
        out = functional.leaky_relu(self.last3(out))
        out = self.last4(out)
        return out


class Residual(Module):
    """
        残差层
        引入"短路连接",直接将输入与输出相加，抑制梯度消失问题，允许更深的网络结构
    """

    def __init__(self, i, o):
        super(Residual, self).__init__()
        # 全连接层 实现线性映射
        self.fc = Linear(i, o)
        # 批次归一化层 提高模型的训练速度并增强模型的稳定性
        self.bn = BatchNorm1d(o)
        # ReLU激活函数用于增加模型的非线性能力
        self.relu = ReLU()

    def forward(self, input_):
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class ODEBlockG(nn.Module):
    """ 常微分方程块 """
    def __init__(self, odefunc):
        super(ODEBlockG, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)

        return out[1]


class PixelNorm(nn.Module):
    """ 像素归一化 """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class ODEFuncG(nn.Module):
    """ 常微分方程函数 """
    def __init__(self, first_layer_dim):
        super(ODEFuncG, self).__init__()

        self.dim = first_layer_dim

        self.layer_start = PixelNorm()
        seq = [ nn.Linear(first_layer_dim + 1, first_layer_dim + 1),
                nn.LeakyReLU(0.2) ]
        seq *= 7
        seq.append(nn.Linear(first_layer_dim + 1, first_layer_dim))
        self.layer_t = Sequential(*seq)

        for m in self.layer_t:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):

        out = self.layer_start(x)
        tt = torch.ones_like(x[:,[0]]) * t
        out = torch.cat([out, tt],dim = 1)
        out = self.layer_t(out)
        return out

class Generator(Module):
    """ 生成器 """

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        self.ode = ODEBlockG(ODEFuncG(dim))
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        data = self.seq(input_)
        return data

class OCTGAN(BaseSynthesizerModel):

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
        transformer=None,
        sampler=None,
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
        self.num_split = 3

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)

        self._transformer = transformer
        self._data_sampler = sampler
        self._generator = None

    def odetime(self, num_split):
        return [torch.tensor([1 / num_split * i], dtype=torch.float32, requires_grad=True, device='cpu') for
                    i in range(1, num_split)]

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """


        # 多次尝试对输入的logits进行Gumbel-Softmax采样,如果采样结果中没有NaN值,返回采样结果
        # 否则连续进行10次尝试,如果在这10次尝试后,采样仍然返回NaN,抛出 ValueError错误
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError("gumbel_softmax returning NaN.")

    def _apply_activate(self, data):
        """ 应用合适的激活函数到生成器的输出上 """
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
        """ 计算指定离散列的交叉熵损失 """
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
        """ 确认discrete_column是否存在于train_data
            在训练开始之前对输入数据进行验证，确保给定的离散列名或列索引都是有效的，从而避免在训练过程中出现无法处理的错误.

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

    @random_state
    def fit(self, train_data, discrete_columns: Optional[List] = None, epochs=None):
        if not discrete_columns:
            discrete_columns = []
        # 离散列检查
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
        self._transformer = DataTransformerCTGAN()
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

        all_time = self.odetime(self.num_split)
        optimizerT = optim.Adam(all_time, lr=2e-4, betas=(0.5, 0.9))

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1
        steps_per_epoch = len(train_data) // self._batch_size

        # 开始执行 fit  流程，直到结束
        for i in range(epochs):
            for id_ in range(steps_per_epoch):

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
                    real = self._data_sampler.sample_data(self._batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                real = torch.from_numpy(real.astype('float32')).to(self._device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                # 更新判别器参数
                y_fake = discriminator([fake_cat,all_time])
                y_real = discriminator([fake_cat,all_time])

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, all_time, self._device, self.pac)

                loss_d = loss_d + pen
                optimizerD.zero_grad()
                optimizerT.zero_grad()

                loss_d.backward(retain_graph=True)
                optimizerD.step()
                optimizerT.step()

                # 裁剪时间点t
                with torch.no_grad():
                    for j in range(len(all_time)):
                        if j == 0:
                            start = 0 + 0.00001
                        else:
                            start = all_time[j - 1].item() + 0.00001

                        if j == len(all_time) - 1:
                            end = 1 - 0.00001
                        else:
                            end = all_time[j + 1].item() - 0.00001
                        all_time[j] = all_time[j].clamp_(min=start, max=end)

                # 更新生成器参数
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
                    y_fake = discriminator([torch.cat([fakeact, c1], dim=1), all_time])
                else:
                    y_fake = discriminator([fakeact,all_time])

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):

        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size
            )
        else:
            global_condition_vec = None

        self._generator.eval()
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
        return self._transformer.inverse_transform(data, None)