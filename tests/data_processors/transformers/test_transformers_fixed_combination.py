import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.fixed_combination import (
    FixedCombinationTransformer,
)


@pytest.fixture
def test_fixed_combination_data():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # B is 2 * A
        "C": [5, 5, 5, 5, 5],  # C is constant
        "D": [1, 3, 5, 7, 9],  # D is not related to A or B
        "E": [2, 4, 6, 8, 10],  # E is 2 * A
    }
    df = pd.DataFrame(data)
    yield df


def test_fixed_combination_handling_test_df(test_fixed_combination_data: pd.DataFrame):
    """
    Test the handling of fixed combination columns in a DataFrame.
    This function tests the behavior of a DataFrame when it contains fixed combination columns.
    It is designed to be used in a testing environment, where the DataFrame is passed as an argument.

    Parameters:
    test_fixed_combination_data (pd.DataFrame): The DataFrame to test.

    Returns:
    None

    Raises:
    AssertionError: If the DataFrame does not handle fixed combination columns as expected.
    """

    metadata = Metadata.from_dataframe(test_fixed_combination_data)
    fixed_combinations = metadata.get("fixed_combinations")
    assert fixed_combinations == {
        "A": {"E", "D", "B"},
        "B": {"A", "E", "D"},
        "D": {"A", "E", "B"},
        "E": {"A", "D", "B"},
    }

    # Initialize the FixedCombinationTransformer.
    fixed_combination_transformer = FixedCombinationTransformer()
    # Check if the transformer has not been fitted yet.
    assert fixed_combination_transformer.fitted is False

    # Fit the transformer with the DataFrame.
    fixed_combination_transformer.fit(metadata)

    # Check if the transformer has been fitted after the fit operation.
    assert fixed_combination_transformer.fitted

    # Check the fixed combination columns
    assert fixed_combination_transformer.fixed_combinations == {
        "A": {"E", "D", "B"},
        "B": {"A", "E", "D"},
        "D": {"A", "E", "B"},
        "E": {"A", "D", "B"},
    }

    # Transform the DataFrame using the transformer.
    transformed_df = fixed_combination_transformer.convert(test_fixed_combination_data)

    # 确保所有原始列都被保留
    for column in test_fixed_combination_data.columns:
        assert column in transformed_df.columns, f"列 {column} 应该被保留在转换后的数据中。"

    # 检查转换后的数据是否符合预期
    assert transformed_df.shape == test_fixed_combination_data.shape
