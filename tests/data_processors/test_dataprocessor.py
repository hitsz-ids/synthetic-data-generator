from __future__ import annotations

import pandas as pd
import pytest

from sdgx.data_processors.base import DataProcessor


@pytest.fixture
def single_demo_data_df(demo_single_table_path):
    df = pd.read_csv(demo_single_table_path)
    return df[["workclass", "age", "occupation"]]


@pytest.fixture
def attach_demo_data_df(demo_single_table_path):
    df = pd.read_csv(demo_single_table_path)
    return df[["education", "capital-gain"]]


@pytest.fixture
def base_data_processor():
    d = DataProcessor()
    yield d


def test_remove_columns(single_demo_data_df: pd.DataFrame, base_data_processor: DataProcessor):
    assert "occupation" in single_demo_data_df.columns
    assert "workclass" in single_demo_data_df.columns
    assert "age" in single_demo_data_df.columns
    result_df = base_data_processor.remove_columns(single_demo_data_df, ["workclass", "occupation"])
    assert "occupation" not in result_df.columns
    assert "workclass" not in result_df.columns
    assert "age" in result_df.columns


def test_attach_columns(
    single_demo_data_df: pd.DataFrame,
    attach_demo_data_df: pd.DataFrame,
    base_data_processor: DataProcessor,
):
    assert "occupation" in single_demo_data_df.columns
    assert "workclass" in single_demo_data_df.columns
    assert "age" in single_demo_data_df.columns
    assert "education" not in single_demo_data_df.columns
    assert "capital-gain" not in single_demo_data_df.columns
    assert "education" in attach_demo_data_df.columns
    assert "capital-gain" in attach_demo_data_df.columns
    attached_df = base_data_processor.attach_columns(single_demo_data_df, attach_demo_data_df)
    assert "occupation" in attached_df.columns
    assert "workclass" in attached_df.columns
    assert "age" in attached_df.columns
    assert "education" in attached_df.columns
    assert "capital-gain" in attached_df
