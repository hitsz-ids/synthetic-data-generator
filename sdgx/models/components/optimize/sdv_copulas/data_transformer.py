import numpy as np
import pandas as pd

from sdgx.models.components.sdv_ctgan.data_transformer import (
    ColumnTransformInfo,
    DataTransformer,
    SpanInfo,
)
from sdgx.models.components.sdv_rdt.transformers import ClusterBasedNormalizer
from sdgx.models.components.sdv_rdt.transformers.categorical import FrequencyEncoder

# TODO(Enhance) - Use different type of Encoder for discrete, like ordered columns, high cardinality columns...


class StatisticDataTransformer(DataTransformer):
    """Data Transformer for statistical models like Gaussian Copula."""

    def _fit_continuous(self, data):
        """Train ClusterBasedNormalizer for continuous columns."""
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(model_missing_values=True, max_clusters=1)
        gm.fit(data, column_name)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_info=[SpanInfo(1, "tanh")],
            output_dimensions=1,
        )

    def _transform_continuous(self, column_transform_info, data):
        """Transform continuous column."""
        gm = column_transform_info.transform
        transformed = gm.transform(data)
        return transformed[f"{data.columns[0]}.normalized"].to_numpy().reshape(-1, 1)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        """Inverse transform continuous column."""
        gm = column_transform_info.transform
        column_name = column_transform_info.column_name

        # Create dataframe
        data = pd.DataFrame(
            {
                f"{column_name}.normalized": column_data.flatten(),
                f"{column_name}.component": [0] * len(column_data),  # virtual component
            }
        )

        if sigmas is not None:
            data[f"{column_name}.normalized"] = np.random.normal(
                data[f"{column_name}.normalized"], sigmas[st]
            )

        # Reverse data
        result = gm.reverse_transform(data)

        # Ensure correct column
        if column_name in result.columns:
            return result[column_name]
        else:
            # Try first column
            return result.iloc[:, 0]

    def _fit_discrete(self, data):
        """Fit frequency encoder for discrete column."""
        column_name = data.columns[0]
        freq_encoder = FrequencyEncoder()
        freq_encoder.fit(data, column_name)

        # Save original unique values for inverse transform
        self._discrete_values = (
            {column_name: data[column_name].unique().tolist()}
            if not hasattr(self, "_discrete_values")
            else {**self._discrete_values, column_name: data[column_name].unique().tolist()}
        )

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=freq_encoder,
            output_info=[SpanInfo(1, "tanh")],
            output_dimensions=1,
        )

    def _transform_discrete(self, column_transform_info, data):
        """Transform discrete column using frequency encoding."""
        freq_encoder = column_transform_info.transform
        return freq_encoder.transform(data).to_numpy().reshape(-1, 1)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        """Inverse transform discrete column from frequency encoding."""
        freq_encoder = column_transform_info.transform
        column_name = column_transform_info.column_name

        # Use frequency encoder to reverse transform
        data = pd.DataFrame({column_name: column_data.flatten()})

        # Get all possible category values
        categories = freq_encoder.starts["category"].values

        # Find the closest category for each frequency value
        result = []
        for val in data[column_name]:
            # The index of the closest start point
            starts = freq_encoder.starts.index.values
            idx = np.abs(starts - val).argmin()
            # Set which category does the closest start point belong to
            result.append(categories[idx])

        return pd.Series(result, index=data.index, dtype=freq_encoder.dtype)
