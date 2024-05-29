import numpy as np
import pandas as pd
import pytest

from sdgx.data_processors.transformers.column_order import ColumnOrderTransformer
from sdgx.data_models.metadata import Metadata

@pytest.fixture
def df_data():
    row_cnt = 100
    header = ["int_id", "str_id", "int_random", "bool_random", 'float_random']

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    bool_random = int_random < 50
    float_random = np.random.randn(row_cnt)

    X = [[int_id[i], str_id[i], int_random[i], bool_random[i], float_random[i]] for i in range(row_cnt)]
    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)
    yield df

@pytest.fixture
def df_data_processed():
    """
    A synthetic dataframe after being processed by other processor / model.
    """
    row_cnt = 100
    header = [ "int_random", "int_id",'float_random_2', "bool_random", 'float_random',"bool_random_2", "str_id",]

    int_id = list(range(row_cnt))
    str_id = list("id_" + str(i) for i in range(row_cnt))

    int_random = np.random.randint(100, size=row_cnt)
    bool_random = int_random < 50
    bool_random_2 = int_random < 40
    float_random = np.random.randn(row_cnt)
    float_random_2 = np.random.randn(row_cnt)

    X = [[int_random[i], int_id[i],  float_random_2[i], bool_random[i], float_random[i], bool_random_2[i], str_id[i],] for i in range(row_cnt)]
    # Convert the list of lists to a DataFrame
    df = pd.DataFrame(X, columns=header)
    yield df


def test_numeric_transformer_fit_test_df(df_data: pd.DataFrame, df_data_processed: pd.DataFrame):
    """
    This module contains tests for the order of transformers in a data processing pipeline.
    Each test case verifies that the order of transformers is maintained correctly during the data processing.
    """
    # about the df_data and df_data_processed
    assert df_data.columns.to_list() == ["int_id", "str_id", "int_random", "bool_random", 'float_random']
    assert df_data_processed.columns.to_list() == ["int_random", "int_id",'float_random_2', "bool_random", 'float_random',"bool_random_2", "str_id",]
    
    # shuffled header
    assert df_data.shape == (100, 5)
    assert df_data_processed.shape == (100, 7) # different shape 
    
    # get metadata 
    metadata_df = Metadata.from_dataframe(df_data)
    
    # fit the transformer 
    transformer = ColumnOrderTransformer()
    transformer.fit(metadata_df)
    assert transformer.column_list == ["int_id", "str_id", "int_random", "bool_random", 'float_random']
    
    # convert will not change the data 
    transformed_df = transformer.convert(df_data)
    assert transformed_df.columns.to_list() == df_data.columns.to_list()
    assert transformed_df.shape == (100, 5)
    assert df_data.equals(transformed_df)

    # test that the header / column order of transformed dataframe is equal to the original 
    convert_transformed_df = transformer.reverse_convert(df_data_processed)
    assert df_data.columns.to_list() == convert_transformed_df.columns.to_list()
    assert convert_transformed_df.shape == (100,5)
    assert convert_transformed_df.columns.to_list() == transformer.column_list


