import pandas as pd
import pytest

from sdgx.data_models.inspectors.discrete import DiscreteInspector


@pytest.fixture
def inspector():
    yield DiscreteInspector()


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


def test_inspector(inspector: DiscreteInspector, raw_data):
    inspector.fit(raw_data)
    assert inspector.ready
    assert inspector.discrete_columns
    assert sorted(inspector.inspect()["discrete_columns"]) == sorted(
        [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "gender",
            "native-country",
            "income",
        ]
    )
    assert inspector.inspect_level == 10


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
