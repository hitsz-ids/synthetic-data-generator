from __future__ import annotations

from typing import List, Literal, Union

from sdgx.models.components.sdv_rdt.transformers import (
    ClusterBasedNormalizer,
    NormalizedFrequencyEncoder,
    NormalizedLabelEncoder,
    OneHotEncoder,
)

CategoricalEncoderInstanceType = Union[
    OneHotEncoder, NormalizedFrequencyEncoder, NormalizedLabelEncoder
]
ContinuousEncoderInstanceType = Union[ClusterBasedNormalizer]
TransformerEncoderInstanceType = Union[
    CategoricalEncoderInstanceType, ContinuousEncoderInstanceType
]
ActivationFuncType = Literal["softmax", "tanh", "linear"]
ColumnTransformType = Literal["discrete", "continuous"]


class SpanInfo:
    def __init__(self, dim: int, activation_fn: ActivationFuncType):
        self.dim: int = dim
        self.activation_fn: ActivationFuncType = activation_fn


class ColumnTransformInfo:
    def __init__(
        self,
        column_name: str,
        column_type: ColumnTransformType,
        transform: TransformerEncoderInstanceType,
        output_info: List[SpanInfo],
        output_dimensions: int,
    ):
        self.column_name: str = column_name
        self.column_type: str = column_type
        self.transform: TransformerEncoderInstanceType = transform
        self.output_info: List[SpanInfo] = output_info
        self.output_dimensions: int = output_dimensions
