import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.specific_combination import (
    SpecificCombinationTransformer,
)


@pytest.fixture
def train_data():
    return pd.DataFrame(
        {
            "price_usd": [100, 200, 300],
            "price_cny": [700, 1400, 2100],
            "price_eur": [90, 180, 270],
            "size_cm": [10, 20, 30],
            "size_inch": [3.94, 7.87, 11.81],
            "size_m": [0.1, 0.2, 0.3],
        }
    )


@pytest.fixture
def test_data():
    return pd.DataFrame(
        {
            "price_usd": [200, 200, 100],
            "price_cny": [1400, 1400, 2100],
            "price_eur": [90, 270, 270],
            "size_cm": [10, 20, 20],
            "size_inch": [3.94, 7.87, 11.81],
            "size_m": [0.1, 0.3, 0.3],
        }
    )


@pytest.fixture
def expected_data():
    return pd.DataFrame(
        {
            "price_usd": [200, 200, 300],
            "price_cny": [1400, 1400, 2100],
            "price_eur": [180, 180, 270],
            "size_cm": [10, 20, 30],
            "size_inch": [3.94, 7.87, 11.81],
            "size_m": [0.1, 0.2, 0.3],
        }
    )


def test_specific_combination_transformer(train_data, test_data, expected_data):
    transformer = SpecificCombinationTransformer()
    metadata = Metadata.from_dataframe(train_data)

    # Turn parameter into a tuple
    metadata.update(
        {
            "specific_combinations": {
                ("price_usd", "price_cny", "price_eur"),
                ("size_cm", "size_inch", "size_m"),
            }
        }
    )
    transformer.fit(metadata=metadata, tabular_data=train_data)
    result = transformer.reverse_convert(test_data)
    pd.testing.assert_frame_equal(result, expected_data, check_dtype=False)
