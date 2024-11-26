"""DataTransformer module."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import autonotebook as tqdm

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata, CategoricalEncoderType
from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader
from sdgx.models.components.sdv_rdt.transformers import (
    ClusterBasedNormalizer,
    OneHotEncoder,
)
from sdgx.models.components.sdv_rdt.transformers.categorical import LabelEncoder, NormalizedLabelEncoder
from sdgx.utils import logger

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    ["column_name", "column_type", "transform", "output_info", "output_dimensions"],
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005, metadata=None):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self.metadata: Metadata = metadata
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=min(len(data), 10))
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_info=[SpanInfo(1, "tanh"), SpanInfo(num_components, "softmax")],  # 贝叶斯gmm，多个正态分布，选择一个
            output_dimensions=1 + num_components,
        )

    def _fit_discrete(self, data, encoder_type: CategoricalEncoderType = None):
        """Fit one hot encoder for discrete column.

        Args:
            data (pd.DataFrame):
                A dataframe containing a column.

        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]

        encoder = OneHotEncoder()
        encoder.fit(data, column_name)
        num_categories = len(encoder.dummies)
        activate_fn = "softmax"

        checked = self.metadata.check_categorical_threshold(num_categories)
        if encoder_type == 'onehot' or not checked:
            pass
        elif encoder_type == 'label':
            encoder = NormalizedLabelEncoder(order_by="alphabetical")
            encoder.fit(data, column_name)
            num_categories = 1
            activate_fn = "liner"
        else:
            raise ValueError("column encoder must be either 'onehot'(default) or 'label'")

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=encoder,
            output_info=[SpanInfo(num_categories, activate_fn)],
            output_dimensions=num_categories,
        )

    def fit(self, data_loader: DataLoader, discrete_columns=()):
        """Fit the ``DataTransformer``.

        Fits a ``ClusterBasedNormalizer`` for continuous columns and a
        ``OneHotEncoder`` for discrete columns.

        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        self._column_raw_dtypes = data_loader[: data_loader.chunksize].infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in tqdm.tqdm(data_loader.columns(), desc="Preparing data", delay=3):
            if column_name in discrete_columns:
                #  or column_name in self.metadata.label_columns
                logger.debug(f"Fitting discrete column {column_name}...")

                column_transform_info = self._fit_discrete(data_loader[[column_name]],
                                                           self.metadata.categorical_encoder[
                                                               column_name] if column_name in self.metadata.categorical_encoder else 'onehot')
            else:
                logger.debug(f"Fitting continuous column {column_name}...")
                column_transform_info = self._fit_continuous(data_loader[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def get_column_transform_info_list(self):
        return self._column_transform_info_list

    def _transform_continuous(self, column_transform_info, data):
        logger.debug(f"Transforming continuous column {column_transform_info.column_name}...")
        column_name = data.columns[0]
        data[column_name] = data[column_name].to_numpy().flatten()
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        logger.debug(f"Transforming discrete column {column_transform_info.column_name}...")
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, raw_data, column_transform_info_list) -> NDArrayLoader:
        """Take a Pandas DataFrame and transform columns synchronous.

        Outputs a list with Numpy arrays.
        """
        loader = NDArrayLoader(save_to_file=False)
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            if column_transform_info.column_type == "continuous":
                loader.store(self._transform_continuous(column_transform_info, data).astype(float))
            else:
                loader.store(self._transform_discrete(column_transform_info, data).astype(float))

        return loader

    def _parallel_transform(self, raw_data, column_transform_info_list) -> NDArrayLoader:
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == "continuous":
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        p = Parallel(n_jobs=-1, return_as="generator")

        loader = NDArrayLoader(save_to_file=False)
        for ndarray in tqdm.tqdm(
                p(processes), desc="Transforming data", total=len(processes), delay=3
        ):
            loader.store(ndarray.astype(float))
        return loader

    def transform(self, dataloader: DataLoader) -> NDArrayLoader:
        """Take raw data and output a matrix data."""

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if dataloader.shape[0] < 500:
            loader = self._synchronous_transform(dataloader, self._column_transform_info_list)
        else:
            loader = self._parallel_transform(dataloader, self._column_transform_info_list)

        return loader

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data = data.astype(float)
        data.iloc[:, 1] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """

        # 这里可以考虑并行化或者 apply TODO
        st = 0
        recovered_column_data_list = []
        column_names = []

        for column_transform_info in tqdm.tqdm(
                self._column_transform_info_list, desc="Inverse transforming", delay=3
        ):
            dim = column_transform_info.output_dimensions
            column_data = data[:, st: st + dim]
            if column_transform_info.column_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )
            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()
        print("Recovered {} samples".format(len(recovered_data)))
        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot),
        }
