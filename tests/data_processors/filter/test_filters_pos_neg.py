import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.filter.positive_negative import PositiveNegativeFilter


@pytest.fixture
def pos_neg_test_df():
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


def test_positive_negative_filter(pos_neg_test_df: pd.DataFrame):
    # Get metadata
    metadata_df = Metadata.from_dataframe(pos_neg_test_df)

    # Initialize PositiveNegativeFilter
    pos_neg_filter = PositiveNegativeFilter()
    assert not pos_neg_filter.fitted

    # Test fit method
    pos_neg_filter.fit(metadata_df)
    assert pos_neg_filter.fitted
    assert pos_neg_filter.positive_columns == {"int_id", "pos_int", "pos_float"}
    assert pos_neg_filter.negative_columns == {"neg_int", "neg_float"}

    # Test convert method
    converted_df = pos_neg_filter.convert(pos_neg_test_df)
    assert converted_df.shape == pos_neg_test_df.shape
    assert (converted_df["pos_int"] >= 0).all()
    assert (converted_df["pos_float"] >= 0).all()
    assert (converted_df["neg_int"] <= 0).all()
    assert (converted_df["neg_float"] <= 0).all()

    # Test reverse_convert method
    reverse_converted_df = pos_neg_filter.reverse_convert(converted_df)
    assert reverse_converted_df.shape[1] == converted_df.shape[1]
    assert (reverse_converted_df["pos_int"] >= 0).all()
    assert (reverse_converted_df["pos_float"] >= 0).all()
    assert (reverse_converted_df["neg_int"] <= 0).all()
    assert (reverse_converted_df["neg_float"] <= 0).all()

    # Check: whether mixed columns remained unchanged
    pd.testing.assert_series_equal(pos_neg_test_df["mixed_int"], reverse_converted_df["mixed_int"])
    pd.testing.assert_series_equal(
        pos_neg_test_df["mixed_float"], reverse_converted_df["mixed_float"]
    )

    # Check if reverse_convert correctly filtered out non-compliant rows (samples)
    assert reverse_converted_df.shape[0] <= pos_neg_test_df.shape[0]


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
