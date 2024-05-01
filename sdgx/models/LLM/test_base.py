import pytest

from sdgx.models.LLM.base import LLMBaseModel


def test_form_columns_description_int():
    llm_base_model = LLMBaseModel()
    sampled_data = {"int_column_example": [0, 1, 2, 3, 4, 5]}
    expected = (
        "\ncolumn #0\ncolumn name: int_column_example\ncolumn data type: int64\nmin value: 0\n"
        "max value: 5\nmean value: 2.5\nstandard deviation: 1.8708286933869707\n"
    )
    assert llm_base_model.Form_columns_description(sampled_data) == expected


def test_form_columns_description_object():
    llm_base_model = LLMBaseModel()
    sampled_data = {"object_column_example": ["object 1", "object 2", "object 3"]}
    expected = (
        "\ncolumn #0\ncolumn name: object_column_example\ncolumn data type: object\n"
        "number of all object values: 3\nnumber of unique object values: 3\n"
    )
    assert llm_base_model.Form_columns_description(sampled_data) == expected
