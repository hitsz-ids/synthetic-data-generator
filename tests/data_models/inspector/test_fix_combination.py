import pandas as pd
import pytest

from sdgx.data_models.inspectors.fix_combination import FixCombinationInspector

@pytest.fixture
def test_fix_combination_data():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [2, 4, 6, 8, 10],  # B is 2 * A
        "C": [5, 5, 5, 5, 5],   # C is constant
        "D": [1, 3, 5, 7, 9],   # D is not related to A or B
        "E": [2, 4, 6, 8, 10],  # B is 2 * A
    }
    df = pd.DataFrame(data)
    yield df

def test_fix_combination_inspector(test_fix_combination_data: pd.DataFrame):
    inspector = FixCombinationInspector()
    inspector.fit(test_fix_combination_data)
    assert inspector.ready
    assert inspector.fixed_combinations

    expected_combinations = {'A': {'E', 'D', 'B'}, 'B': {'A', 'E', 'D'}, 'D': {'A', 'E', 'B'}, 'E': {'A', 'D', 'B'}}
    assert inspector.fixed_combinations == expected_combinations
    assert inspector.inspect_level == 70

if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])