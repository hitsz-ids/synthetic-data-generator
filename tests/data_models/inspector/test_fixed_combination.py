import pandas as pd
import pytest

from sdgx.data_models.inspectors.fixed_combination import FixedCombinationInspector


@pytest.fixture
def test_fixed_combination_data():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # B is 2 * A
        "C": [5, 5, 5, 5, 5],  # C is constant
        "D": [1, 3, 5, 7, 9],  # D is not related to A or B
        "E": [2, 4, 6, 8, 10],  # B is 2 * A
        "categorical_1": ["apple", "banana", "apple", "banana", "cherry"],  # fruits
        "categorical_2": ["red", "yellow", "red", "yellow", "pink"],  # colors
        "categorical_3": [1, 2, 1, 2, 3],  # int to string
        "categorical_4": ["one", "two", "one", "two", "three"],  # int to string
        "categorical_5": [0.1, 0.5, 0.1, 0.5, 1.0],  # float to string
        "categorical_6": ["light", "medium", "light", "medium", "heavy"],  # weight descriptors
    }
    df = pd.DataFrame(data)
    yield df


def test_fixed_combination_inspector(test_fixed_combination_data: pd.DataFrame):
    inspector = FixedCombinationInspector()
    inspector.fit(test_fixed_combination_data)
    assert inspector.ready
    assert inspector.fixed_combinations

    expected_combinations = {
        "A": {"categorical_3", "D", "E", "B"},
        "B": {"categorical_3", "D", "E", "A"},
        "D": {"categorical_3", "E", "A", "B"},
        "E": {"categorical_3", "D", "A", "B"},
        "categorical_3": {"categorical_4", "D", "E", "A", "B"},
        "categorical_1": {"categorical_2"},
        "categorical_5": {"categorical_6"},
    }

    assert inspector.fixed_combinations == expected_combinations
    assert inspector.inspect_level == 70


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
