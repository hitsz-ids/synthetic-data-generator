from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from packaging import version
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
from tqdm import autonotebook as tqdm

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader
from sdgx.models.components.optimize.sdv_ctgan.data_sampler import DataSampler
from sdgx.models.components.optimize.sdv_ctgan.data_transformer import DataTransformer
from sdgx.models.components.sdv_ctgan.synthesizers.base import (
    BaseSynthesizer as SDVBaseSynthesizer,
)
from sdgx.models.components.sdv_ctgan.synthesizers.base import random_state
from sdgx.models.extension import hookimpl
from sdgx.models.ml.single_table.base import MLSynthesizerModel
from sdgx.utils import logger


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


class CTGANSynthesizerModel(MLSynthesizerModel, SDVBaseSynthesizer):
    """
    Modified from ``sdgx.models.components.sdv_ctgan.synthesizers.ctgan.CTGANSynthesizer``.
    A CTGANSynthesizer but provided :ref:`SynthesizerModel` interface with chunked fit.

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
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        device (str):
            Device to run the training on. Preferred to be 'cuda' for GPU if available.
    """

    MODEL_SAVE_NAME = "ctgan.pkl"

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
        epochs=300,
        pac=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
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
        self._epochs = epochs
        self.pac = pac

        self._device = torch.device(device)

        # Following components are initialized in `_pre_fit`
        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._ndarry_loader = None
        self.data_dim = None

    def fit(self, metadata: Metadata, dataloader: DataLoader, epochs=None, *args, **kwargs):
        # In the future, sdgx use `sdgx.data_processor.transformers.discrete` to handle discrete_columns
        # the original sdv transformer will be removed in version 0.3.0
        # This will be done in another PR.
        discrete_columns = list(metadata.get("discrete_columns"))
        if epochs is not None:
            self._epochs = epochs
        self._pre_fit(dataloader, discrete_columns)
        if self.fit_data_empty:
            logger.info("CTGAN fit finished because of empty df detected.")
            return
        logger.info("CTGAN prefit finished, start CTGAN training.")
        self._fit(len(self._ndarry_loader))
        logger.info("CTGAN training finished.")

    def _pre_fit(self, dataloader: DataLoader, discrete_columns: list[str] = None) -> NDArrayLoader:
        if not discrete_columns:
            discrete_columns = []

        # self._validate_discrete_columns(dataloader.columns(), discrete_columns)
        discrete_columns = self._filter_discrete_columns(dataloader.columns(), discrete_columns)
        # if the df is empty, we don't need to do anything
        if self.fit_data_empty:
            return
        # Fit Transformer and DataSampler
        self._transformer = DataTransformer()
        logger.info("Fitting model's transformer...")
        self._transformer.fit(dataloader, discrete_columns)
        logger.info("Transforming data...")
        self._ndarry_loader = self._transformer.transform(dataloader)

        logger.info("Sampling data.")
        self._data_sampler = DataSampler(
            self._ndarry_loader, self._transformer.output_info_list, self._log_frequency
        )

        logger.info("Initialize Generator.")
        # Initialize Generator
        self.data_dim = self._transformer.output_dimensions
        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            self.data_dim,
        ).to(self._device)

    @random_state
    def _fit(self, data_size: int):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            dataloader: :ref:`DataLoader` for the training data processed by :ref:`DataProcessor`.

        """
        logger.info(f"Fit using data_size:{data_size}, data_dim: {self.data_dim}.")
        epochs = self._epochs
        # data_dim = self._transformer.output_dimensions
        discriminator = Discriminator(
            self.data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

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

        logger.info("Starting model training, epochs: {}".format(epochs))
        steps_per_epoch = max(data_size // self._batch_size, 1)
        for i in range(epochs):
            start_time = time.time()
            for id_ in tqdm.tqdm(range(steps_per_epoch), desc="Fitting batches", delay=3):
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

                    optimizerD.zero_grad()
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

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            logger.info(
                f"Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},"  # noqa: T001
                f" Loss D: {loss_d.detach().cpu(): .4f},"
                f" Time: {time.time() - start_time: .4f}",
            )

    def sample(self, count: int, *args, **kwargs) -> pd.DataFrame:
        if self.fit_data_empty:
            return pd.DataFrame(index=range(count))
        return self._sample(count, *args, **kwargs)

    @random_state
    def _sample(self, n, condition_column=None, condition_value=None):
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
        for i in tqdm.tqdm(range(steps), desc="Sampling batches", delay=3):
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

    def save(self, save_dir: str | Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        return SDVBaseSynthesizer.save(self, save_dir / self.MODEL_SAVE_NAME)

    @classmethod
    def load(cls, save_dir: str | Path) -> "CTGANSynthesizerModel":
        return SDVBaseSynthesizer.load(save_dir / cls.MODEL_SAVE_NAME)

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
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(
                    logits, tau=tau, hard=hard, eps=eps, dim=dim
                )
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

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

    def _filter_discrete_columns(self, train_data: List[str], discrete_columns: List[str]):
        """
        We filter PII Column here, which PII would only be discrete for now.
        As PII would be generating from PII Generator which not synthetic from model.

        Besides we need to figure it out when to stop model fitting:
        The original data consists entirely of discrete column data, and all of this discrete column data is PII.

        For `train_data`, there are three possibilities for the columns type.
         - train_data = valid_discrete + valid_continue
         - train_data = valid_continue
         - train_data = valid_discrete

        For `discrete_columns`, discrete_columns = invalid_discrete(PII) + valid_discrete

        Thus, valid_discrete = discrete_columns - invalid_discrete
                             = discrete_columns - Set.intersection(train_data, discrete_columns)

        Thus, original_data_is_all_PII: discrete_columns is not empty & train_data is empty
        """
        # Discrete_columns is empty - simple an empty list, but we need to continue fitting continue columns
        if len(discrete_columns) == 0:
            return discrete_columns

        # Discrete_columns is not empty - check if train_data is empty for stop model fitting
        if len(train_data) == 0:
            self.fit_data_empty = True
            return discrete_columns

        # Filter valid discrete columns
        invalid_columns = set(discrete_columns) - set(train_data)
        return set(discrete_columns) - set(invalid_columns)

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame or list):
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
        elif isinstance(train_data, list):
            invalid_columns = set(discrete_columns) - set(train_data)
        else:
            raise TypeError("``train_data`` should be either pd.DataFrame or np.array.")

        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)


@hookimpl
def register(manager):
    manager.register("CTGAN", CTGANSynthesizerModel)
