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
        "categorical_one": [
            "co1",
            "co3",
            "co2",
            "co9",
            "co1",
        ],  # categorical_one and categorical_two have a one-to-one correspondence.
        "categorical_two": ["ct1", "ct3", "ct2", "ct9", "ct1"],
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
        "categorical_one": {"categorical_two"},
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
        "categorical_one": {"categorical_two"},
    }

    # Transform the DataFrame using the transformer.
    transformed_df = fixed_combination_transformer.convert(test_fixed_combination_data)

    # Ensure all original columns are retained
    for column in test_fixed_combination_data.columns:
        assert (
            column in transformed_df.columns
        ), f"Column {column} should be retained in the transformed data."

    # Check if the transformed data meets expectations
    assert transformed_df.shape == test_fixed_combination_data.shape


def test_categorical_fixed_combinations(test_fixed_combination_data):
    """Test the fixed combination relationship of categorical variables"""
    # 测试分类变量的固定组合关系
    metadata = Metadata.from_dataframe(test_fixed_combination_data)
    transformer = FixedCombinationTransformer()
    transformer.fit(metadata)

    # Verify that the correspondence between categorical_one and categorical_two is detected
    # 验证categorical_one和categorical_two的对应关系被检测到
    assert "categorical_one" in transformer.fixed_combinations
    assert "categorical_two" in transformer.fixed_combinations["categorical_one"]

    # Verify that the transformed data maintains the original correspondence
    # 验证转换后的数据保持原有的对应关系
    transformed_df = transformer.convert(test_fixed_combination_data)
    assert all(
        transformed_df["categorical_one"].map(
            dict(
                zip(
                    test_fixed_combination_data["categorical_one"],
                    test_fixed_combination_data["categorical_two"],
                )
            )
        )
        == transformed_df["categorical_two"]
    )
