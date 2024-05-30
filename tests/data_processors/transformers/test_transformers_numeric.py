import numpy as np
import pandas as pd
import pytest

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.transformers.numeric import NumericValueTransformer


@pytest.fixture
def df_data():
    row_cnt = 1000
    header = ["int_id", "str_id", "int_random", "bool_random", "float_random"]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    bool_random = int_random < 50
    float_random = np.random.randn(row_cnt)

    X = [
        [int_id[i], str_id[i], int_random[i], bool_random[i], float_random[i]]
        for i in range(row_cnt)
    ]
    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)
    yield df


def calculate_mean_and_variance(df, numeric_df):
    if not isinstance(numeric_df, list):
        raise ValueError("numeric_df should be a list of column names.")
    for col in numeric_df:
        if col not in df.columns:
            raise ValueError(f"Column {col} does not exist in the DataFrame.")
    stats = {}
    for col in numeric_df:
        mean = df[col].mean()
        variance = df[col].var()
        stats[col] = {"mean": mean, "variance": variance}
    return stats


def test_numeric_transformer_fit_test_df(df_data: pd.DataFrame):
    """ """
    # get metadata
    metadata_df = Metadata.from_dataframe(df_data)

    # fit the transformer
    transformer = NumericValueTransformer()
    transformer.fit(metadata_df, df_data)
    assert transformer.int_columns == {"int_random", "int_id"}
    assert transformer.float_columns == {"float_random"}


def test_numeric_transformer_convert_test_df(df_data: pd.DataFrame):
    """ """
    # get metadata
    metadata_df = Metadata.from_dataframe(df_data)
    # fit the transformer
    transformer = NumericValueTransformer()
    transformer.fit(metadata_df, df_data)
    # convert the data
    converted_df = transformer.convert(df_data)
    numerical_columns = list(transformer.int_columns) + list(transformer.float_columns)
    converted_status = calculate_mean_and_variance(converted_df, numerical_columns)
    assert type(converted_df) == pd.DataFrame
    assert converted_df.shape == df_data.shape
    assert np.isclose(converted_status["int_id"]["mean"], 0.0)
    assert np.isclose(converted_status["int_random"]["mean"], 0.0)
    assert np.isclose(converted_status["float_random"]["mean"], 0.0)
    assert np.isclose(converted_status["int_id"]["variance"], 1, atol=0.001)
    assert np.isclose(converted_status["int_random"]["variance"], 1, atol=0.001)
    assert np.isclose(converted_status["float_random"]["variance"], 1, atol=0.001)


def test_numeric_transformer_reverse_convert_test_df(df_data: pd.DataFrame):
    """ """
    # fit the transformer
    transformer = NumericValueTransformer()
    transformer.fit(Metadata.from_dataframe(df_data), df_data)
    numerical_columns = list(transformer.int_columns) + list(transformer.float_columns)
    # convert the data
    converted_df = transformer.convert(df_data)
    # invert the converted data
    reverse_converted_df = transformer.reverse_convert(converted_df)
    reverse_converted_status = calculate_mean_and_variance(reverse_converted_df, numerical_columns)
    original_status = calculate_mean_and_variance(df_data, numerical_columns)
    assert type(reverse_converted_df) == pd.DataFrame
    assert reverse_converted_df.shape == df_data.shape
    assert np.isclose(reverse_converted_status["int_id"]["mean"], original_status["int_id"]["mean"])
    assert np.isclose(
        reverse_converted_status["int_random"]["mean"], original_status["int_random"]["mean"]
    )
    assert np.isclose(
        reverse_converted_status["float_random"]["mean"], original_status["float_random"]["mean"]
    )
    assert np.isclose(
        reverse_converted_status["int_id"]["variance"], original_status["int_id"]["variance"]
    )
    assert np.isclose(
        reverse_converted_status["int_random"]["variance"],
        original_status["int_random"]["variance"],
    )
    assert np.isclose(
        reverse_converted_status["float_random"]["variance"],
        original_status["float_random"]["variance"],
    )
