import numpy as np
import pandas as pd
import pytest

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer


@pytest.fixture
def demo_single_table_data_pos_neg():
    row_cnt = 1000
    header = ["int_id", "pos_int", "neg_int", "pos_float", "neg_float", "mixed_int", "mixed_float"]

    np.random.seed(42)
    int_id = list(range(row_cnt))
    pos_int = np.random.randint(1, 100, size=row_cnt)
    neg_int = np.random.randint(-100, 0, size=row_cnt)
    pos_float = np.random.uniform(0, 100, size=row_cnt)
    neg_float = np.random.uniform(-100, 0, size=row_cnt)
    mixed_int = np.random.randint(-50, 50, size=row_cnt)
    mixed_float = np.random.uniform(-50, 50, size=row_cnt)

    X = [
        [
            int_id[i],
            pos_int[i],
            neg_int[i],
            pos_float[i],
            neg_float[i],
            mixed_int[i],
            mixed_float[i],
        ]
        for i in range(row_cnt)
    ]

    yield pd.DataFrame(X, columns=header)


@pytest.fixture
def demo_single_table_data_pos_neg_metadata(demo_single_table_data_pos_neg):
    yield Metadata.from_dataframe(demo_single_table_data_pos_neg)


@pytest.fixture
def demo_single_table_data_pos_neg_path(tmp_path, demo_single_table_data_pos_neg):
    df = demo_single_table_data_pos_neg
    save_path = tmp_path / "dummy_demo_single_table_data_pos_neg.csv"
    df.to_csv(save_path, index=False, header=True)
    yield save_path
    save_path.unlink()


@pytest.fixture
def demo_single_table_data_pos_neg_connector(demo_single_table_data_pos_neg_path):
    yield CsvConnector(
        path=demo_single_table_data_pos_neg_path,
    )


@pytest.fixture
def demo_single_table_data_pos_neg_loader(demo_single_table_data_pos_neg_connector, cacher_kwargs):
    d = DataLoader(demo_single_table_data_pos_neg_connector, cacher_kwargs=cacher_kwargs)
    yield d
    d.finalize()


@pytest.fixture
def ctgan_synthesizer(demo_single_table_data_pos_neg_connector):
    yield Synthesizer(
        model=CTGANSynthesizerModel(epochs=1),
        data_connector=demo_single_table_data_pos_neg_connector,
    )


def test_ctgan_synthesizer_with_pos_neg(
    ctgan_synthesizer: Synthesizer,
    demo_single_table_data_pos_neg_metadata,
    demo_single_table_data_pos_neg_loader,
    demo_single_table_data_pos_neg,
):
    original_data = demo_single_table_data_pos_neg
    metadata = demo_single_table_data_pos_neg_metadata

    # Train the CTGAN model
    ctgan_synthesizer.fit(demo_single_table_data_pos_neg_metadata)
    sampled_data = ctgan_synthesizer.sample(1000)

    # Check each column
    for column in original_data.columns:
        # Skip columns that are identifiers or not intended for positivity checks
        if column == "int_id":
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
