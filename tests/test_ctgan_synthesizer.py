from typing import List

import faker
import numpy as np
import pandas as pd
import pytest

from sdgx.data_connectors.dataframe_connector import DataFrameConnector
from sdgx.data_models.metadata import Metadata
from sdgx.models.components.optimize.sdv_ctgan.data_transformer import (
    DataTransformer,
    SpanInfo,
)
from sdgx.models.components.sdv_rdt.transformers.categorical import (
    NormalizedFrequencyEncoder,
    NormalizedLabelEncoder,
    OneHotEncoder,
)
from sdgx.models.components.sdv_rdt.transformers.numerical import ClusterBasedNormalizer
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer


@pytest.fixture
def demo_single_table_data_pos_neg():
    row_cnt = 1000  # must be 200 multiply because of the encode setting
    np.random.seed(42)
    faker.Faker.seed(42)
    fake = faker.Faker()
    X = {
        "int_id": list(range(row_cnt)),
        "pos_int": np.random.randint(1, 100, size=row_cnt),
        "neg_int": np.random.randint(-100, 0, size=row_cnt),
        "pos_float": np.random.uniform(0, 100, size=row_cnt),
        "neg_float": np.random.uniform(-100, 0, size=row_cnt),
        "mixed_int": np.random.randint(-50, 50, size=row_cnt),
        "mixed_float": np.random.uniform(-50, 50, size=row_cnt),
        "cat_onehot": [str(i) for i in range(row_cnt)],
        "cat_label": [str(i) for i in range(row_cnt)],
        "cat_date": [fake.date() for _ in range(row_cnt)],
        "cat_freq": [str(i) for i in range(row_cnt)],
        "cat_thres_freq": [str(i) for i in range(100)] * (row_cnt // 100),
        "cat_thres_label": [str(i) for i in range(200)] * (row_cnt // 200),
    }
    header = X.keys()
    yield pd.DataFrame(X, columns=list(header))


@pytest.fixture
def demo_single_table_data_pos_neg_metadata(demo_single_table_data_pos_neg):
    metadata = Metadata.from_dataframe(demo_single_table_data_pos_neg.copy(), check=True)
    metadata.categorical_encoder = {
        "cat_onehot": "onehot",
        "cat_label": "label",
        "cat_freq": "frequency",
    }
    metadata.datetime_format = {"cat_date": "%Y-%m-%d"}
    metadata.categorical_threshold = {99: "frequency", 199: "label"}
    yield metadata


@pytest.fixture
def demo_single_table_data_pos_neg_connector(demo_single_table_data_pos_neg):
    yield DataFrameConnector(df=demo_single_table_data_pos_neg)


@pytest.fixture
def ctgan_synthesizer(
    demo_single_table_data_pos_neg_connector, demo_single_table_data_pos_neg_metadata
):
    yield Synthesizer(
        metadata=demo_single_table_data_pos_neg_metadata,
        model=CTGANSynthesizerModel(epochs=1),
        data_connector=demo_single_table_data_pos_neg_connector,
    )


def test_ctgan_synthesizer_with_pos_neg(
    ctgan_synthesizer: Synthesizer, demo_single_table_data_pos_neg
):
    original_data = demo_single_table_data_pos_neg

    # Train the CTGAN model
    ctgan_synthesizer.fit()
    ctgan: CTGANSynthesizerModel = ctgan_synthesizer.model
    transformer: DataTransformer = ctgan._transformer
    transform_list = transformer._column_transform_info_list
    transformed_data = ctgan._ndarry_loader.get_all()

    current_dim = 0
    for item in transform_list:
        span_info: List[SpanInfo] = item.output_info
        col_dim = item.output_dimensions
        current_data = transformed_data[:, current_dim : current_dim + col_dim]
        current_dim += col_dim
        col = item.column_name
        if col in ["cat_freq", "cat_thres_freq"]:
            assert isinstance(item.transform, NormalizedFrequencyEncoder)
            assert col_dim == 1
            assert len(span_info) == 1
            assert span_info[0].activation_fn == "liner"
            assert len(item.transform.intervals) == original_data[col].nunique(dropna=False)
            assert (current_data >= -1).all() and (current_data <= 1).all()
        elif col in ["cat_thres_label", "cat_label"]:
            assert isinstance(item.transform, NormalizedLabelEncoder)
            assert col_dim == 1
            assert len(span_info) == 1
            assert span_info[0].activation_fn == "liner"
            assert len(item.transform.categories_to_values.keys()) == original_data[col].nunique(
                dropna=False
            )
            assert (current_data >= -1).all() and (current_data <= 1).all()
        elif col in ["cat_onehot"]:
            assert isinstance(item.transform, OneHotEncoder)
            nunique = original_data[col].nunique(dropna=False)
            assert col_dim == nunique
            assert len(span_info) == 1
            assert span_info[0].activation_fn == "softmax"
            assert len(item.transform.dummies) == nunique
            assert np.all((current_data == 0) | (current_data == 1))
        else:
            assert isinstance(item.transform, ClusterBasedNormalizer)

    sampled_data = ctgan_synthesizer.sample(1000)

    # Check each column
    for column in original_data.columns:
        # Skip columns that are identifiers or not intended for positivity checks
        if column in ["int_id"] or column.startswith("cat_"):
            continue

        is_all_positive = (original_data[column] >= 0).all()
        is_all_negative = (original_data[column] <= 0).all()

        if is_all_positive:
            # Assert that the sampled_data column is also all positive
            assert (
                sampled_data[column] >= 0
            ).all(), f"Column '{column}' in sampled data should be all positive."
        elif is_all_negative:
            # Assert that the sampled_data column is also all negative
            assert (
                sampled_data[column] <= 0
            ).all(), f"Column '{column}' in sampled data should be all negative."


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
