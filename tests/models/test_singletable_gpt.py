from __future__ import annotations

import pandas as pd
import pytest

from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.exceptions import InitializationError
from sdgx.models.LLM.single_table.gpt import SingleTableGPTModel


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path).head(100)


@pytest.fixture
def single_table_gpt_model():
    yield SingleTableGPTModel()


# When reading the code, please collapse this list
# COLLAPSE ME
gpt_response_list = [
    """
Here are 20 similar data entries generated based on the provided information:

sample 21: relationship is Husband, fnlwgt is 145441, educational-num is 9, education is HS-grad, occupation is Exec-managerial, gender is Male, race is White, workclass is Private, capital-gain is 0, native-country is United-States, marital-status is Married-civ-spouse, income is >50K, age is 40, capital-loss is 1485, hours-per-week is 40
sample 22: income is <=50K, gender is Male, education is Assoc-acdm, native-country is ?, educational-num is 12, hours-per-week is 7, occupation is Prof-specialty, capital-gain is 0, capital-loss is 0, fnlwgt is 154164, race is White, workclass is Private, age is 66, relationship is Not-in-family, marital-status is Never-married
sample 23: relationship is Own-child, marital-status is Never-married, occupation is Transport-moving, income is <=50K, race is White, workclass is Private, hours-per-week is 30, capital-gain is 0, native-country is United-States, capital-loss is 0, fnlwgt is 283499, gender is Male, education is Some-college, age is 20, educational-num is 10
sample 24: marital-status is Married-civ-spouse, gender is Male, capital-gain is 0, capital-loss is 0, educational-num is 9, hours-per-week is 40, fnlwgt is 170772, income is <=50K, occupation is Other-service, relationship is Husband, race is White, education is HS-grad, age is 30, native-country is United-States, workclass is Local-gov
sample 25: native-country is United-States, income is <=50K, capital-gain is 0, hours-per-week is 40, fnlwgt is 367306, educational-num is 10, capital-loss is 0, race is White, relationship is Own-child, gender is Female, occupation is Tech-support, workclass is Private, age is 25, marital-status is Never-married, education is Some-college
sample 26: occupation is Exec-managerial, fnlwgt is 81973, native-country is United-States, capital-loss is 0, educational-num is 14, relationship is Husband, marital-status is Married-civ-spouse, workclass is Private, capital-gain is 0, gender is Male, race is Asian-Pac-Islander, hours-per-week is 40, age is 59, income is >50K, education is Masters
sample 27: gender is Male, fnlwgt is 287268, workclass is Private, native-country is United-States, capital-loss is 0, age is 28, educational-num is 10, hours-per-week is 35, income is <=50K, relationship is Not-in-family, occupation is Other-service, capital-gain is 0, race is White, education is Some-college, marital-status is Never-married
sample 28: race is White, income is >50K, native-country is United-States, workclass is Private, capital-loss is 0, occupation is Craft-repair, hours-per-week is 60, educational-num is 13, gender is Male, fnlwgt is 176729, relationship is Husband, marital-status is Married-civ-spouse, capital-gain is 0, age is 25, education is Bachelors
sample 29: capital-loss is 0, native-country is Cambodia, educational-num is 9, marital-status is Married-civ-spouse, workclass is Self-emp-not-inc, education is HS-grad, age is 42, occupation is Farming-fishing, race is Asian-Pac-Islander, income is >50K, gender is Male, hours-per-week is 40, fnlwgt is 303044, capital-gain is 0, relationship is Husband
sample 30: age is 17, capital-loss is 0, relationship is Own-child, educational-num is 6, workclass is Private, income is <=50K, native-country is United-States, hours-per-week is 15, race is White, occupation is Other-service, fnlwgt is 202521, education is 10th, gender is Male, capital-gain is 0, marital-status is Never-married
sample 31: fnlwgt is 173858, occupation is Exec-managerial, education is Bachelors, native-country is China, hours-per-week is 35, marital-status is Married-civ-spouse, relationship is Husband, age is 38, capital-gain is 7688, workclass is Private, race is Asian-Pac-Islander, income is >50K, capital-loss is 0, gender is Male, educational-num is 13
sample 32: age is 74, workclass is Private, marital-status is Widowed, income is <=50K, hours-per-week is 40, occupation is Priv-house-serv, education is Assoc-voc, gender is Female, race is White, relationship is Not-in-family, fnlwgt is 68326, capital-loss is 0, native-country is United-States, capital-gain is 0, educational-num is 11
sample 33: gender is Male, race is White, age is 28, capital-gain is 0, marital-status is Married-civ-spouse, hours-per-week is 40, relationship is Husband, education is HS-grad, educational-num is 9, income is <=50K, workclass is Private, capital-loss is 0, fnlwgt is 66095, occupation is Sales, native-country is United-States
sample 34: education is Bachelors, occupation is Adm-clerical, race is White, marital-status is Married-civ-spouse, capital-gain is 7688, gender is Male, income is >50K, relationship is Husband, workclass is Private, fnlwgt is 206814, age is 58, hours-per-week is 50, capital-loss is 0, native-country is United-States, educational-num is 13
sample 35: age is 44, occupation is Craft-repair, capital-loss is 0, workclass is Federal-gov, income is >50K, fnlwgt is 243636, capital-gain is 0, education is Assoc-voc, relationship is Husband, educational-num is 11, race is White, marital-status is Married-civ-spouse, hours-per-week is 40, native-country is United-States, gender is Male
sample 36: education is Masters, fnlwgt is 37070, marital-status is Married-civ-spouse, age is 33, relationship is Husband, capital-gain is 0, educational-num is 14, workclass is State-gov, gender is Male, income is <=50K, occupation is Prof-specialty, capital-loss is 0, race is White, native-country is Canada, hours-per-week is 60
sample 37: workclass is Private, educational-num is 7, education is 11th, income is <=50K, relationship is Not-in-family, gender is Male, hours-per-week is 40, native-country is Puerto-Rico, age is 23, fnlwgt is 224217, occupation is Transport-moving, capital-gain is 0, marital-status is Never-married, capital-loss is 0, race is White
sample 38: hours-per-week is 40, capital-gain is 0, gender is Male, marital-status is Never-married, age is 25, capital-loss is 0, native-country is ?, fnlwgt is 310864, income is <=50K, educational-num is 13, race is Black, workclass is Private, education is Bachelors, relationship is Not-in-family, occupation is Tech-support
sample 39: capital-gain is 0, hours-per-week is 55, capital-loss is 0, age is 30, income is <=50K, education is Bachelors, relationship is Not-in-family, marital-status is Never-married, occupation is Exec-managerial, native-country is United-States, gender is Female, race is White, educational-num is 13, workclass is Private, fnlwgt is 128016
sample 40: hours-per-week is 40, fnlwgt is 174515, capital-loss is 0, marital-status is Widowed, native-country is United-States, capital-gain is 0, age is 40, education is HS-grad, occupation is Machine-op-inspct, educational-num is 9, relationship is Unmarried, race is White, gender is Female, workclass is Private, income is <=50K
""",
    """Based on the provided information, here are 15 synthetic data samples generated:

Sample 0: income is <=50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 36, workclass is Self-emp-inc, education is HS-grad, relationship is Husband, native-country is United-States, educational-num is 9.0, occupation is Farming-fishing, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 80.0, fnlwgt is 48063

Sample 1: income is <=50K, race is White, marital-status is Never-married, gender is Male, age is 34, workclass is Private, education is Masters, relationship is Not-in-family, native-country is United-States, educational-num is 14.0, occupation is Prof-specialty, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 40.0, fnlwgt is 189759

Sample 2: income is <=50K, race is Amer-Indian-Eskimo, marital-status is Never-married, gender is Female, age is 19, workclass is Private, education is HS-grad, relationship is Own-child, native-country is United-States, educational-num is 9.0, occupation is Other-service, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 35.0, fnlwgt is 106183

Sample 3: income is <=50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 42, workclass is Private, education is HS-grad, relationship is Husband, native-country is United-States, educational-num is 9.0, occupation is Craft-repair, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 45.0, fnlwgt is 190910

Sample 4: income is >50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 59, workclass is Private, education is 7th-8th, relationship is Husband, native-country is United-States, educational-num is 4.0, occupation is Craft-repair, capital-loss is 0.0, capital-gain is 5178.0, hours-per-week is 50.0, fnlwgt is 107318

Sample 5: income is >50K, race is White, marital-status is Divorced, gender is Male, age is 51, workclass is Private, education is Some-college, relationship is Not-in-family, native-country is United-States, educational-num is 10.0, occupation is Craft-repair, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 42.0, fnlwgt is 43354

Sample 6: income is <=50K, race is White, marital-status is Separated, gender is Male, age is 46, workclass is Private, education is HS-grad, relationship is Not-in-family, native-country is United-States, educational-num is 9.0, occupation is Transport-moving, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 40.0, fnlwgt is 170338

Sample 7: income is >50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 45, workclass is Self-emp-not-inc, education is Some-college, relationship is Husband, native-country is United-States, educational-num is 10.0, occupation is Sales, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 60.0, fnlwgt is 355978

Sample 8: income is <=50K, race is White, marital-status is Never-married, gender is Male, age is 33, workclass is Private, education is Bachelors, relationship is Not-in-family, native-country is United-States, educational-num is 13.0, occupation is Sales, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 45.0, fnlwgt is 90409

Sample 9: income is <=50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 29, workclass is Private, education is 11th, relationship is Husband, native-country is United-States, educational-num is 7.0, occupation is Other-service, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 40.0, fnlwgt is 103634

Sample 10: income is >50K, race is White, marital-status is Never-married, gender is Male, age is 22, workclass is Private, education is HS-grad, relationship is Not-in-family, native-country is United-States, educational-num is 9.0, occupation is Other-service, capital-loss is 0.0, capital-gain is 14084.0, hours-per-week is 60.0, fnlwgt is 54164

Sample 11: income is <=50K, race is White, marital-status is Never-married, gender is Male, age is 17, workclass is ?, education is 10th, relationship is Own-child, native-country is United-States, educational-num is 6.0, occupation is ?, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 40.0, fnlwgt is 165361

Sample 12: income is <=50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 23, workclass is Private, education is 10th, relationship is Husband, native-country is United-States, educational-num is 6.0, occupation is Farming-fishing, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 40.0, fnlwgt is 306309

Sample 13: income is <=50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 38, workclass is Private, education is HS-grad, relationship is Husband, native-country is United-States, educational-num is 9.0, occupation is Farming-fishing, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 50.0, fnlwgt is 89814

Sample 14: income is >50K, race is White, marital-status is Married-civ-spouse, gender is Male, age is 36, workclass is Local-gov, education is Bachelors, relationship is Husband, native-country is United-States, educational-num is 13.0, occupation is Prof-specialty, capital-loss is 0.0, capital-gain is 0.0, hours-per-week is 40.0, fnlwgt is 403681

""",
    """
marital-status is Married-civ-spouse, relationship is Husband, income is <=50K, age is 45, fnlwgt is 200000, occupation is Exec-managerial, native-country is United-States, hours-per-week is 40, workclass is Private, gender is Male, educational-num is 13, capital-gain is 0, capital-loss is 0, race is White, education is Bachelors
marital-status is Never-married, relationship is Not-in-family, income is <=50K, age is 25, fnlwgt is 150000, occupation is Sales, native-country is United-States, hours-per-week is 35, workclass is Private, gender is Female, educational-num is 10, capital-gain is 0, capital-loss is 0, race is White, education is Some-college
marital-status is Divorced, relationship is Unmarried, income is <=50K, age is 35, fnlwgt is 180000, occupation is Adm-clerical, native-country is United-States, hours-per-week is 30, workclass is Private, gender is Female, educational-num is 12, capital-gain is 0, capital-loss is 0, race is White, education is HS-grad
marital-status is Married-civ-spouse, relationship is Wife, income is >50K, age is 50, fnlwgt is 250000, occupation is Prof-specialty, native-country is United-States, hours-per-week is 45, workclass is Self-emp-not-inc, gender is Female, educational-num is 14, capital-gain is 5000, capital-loss is 0, race is White, education is Masters
marital-status is Married-civ-spouse, relationship is Husband, income is >50K, age is 40, fnlwgt is 220000, occupation is Exec-managerial, native-country is United-States, hours-per-week is 50, workclass is Private, gender is Male, educational-num is 13, capital-gain is 10000, capital-loss is 0, race is White, education is Bachelors
marital-status is Never-married, relationship is Own-child, income is <=50K, age is 18, fnlwgt is 100000, occupation is Other-service, native-country is United-States, hours-per-week is 20, workclass is Private, gender is Male, educational-num is 9, capital-gain is 0, capital-loss is 0, race is Black, education is Some-college
marital-status is Married-civ-spouse, relationship is Wife, income is >50K, age is 35, fnlwgt is 180000, occupation is Prof-specialty, native-country is United-States, hours-per-week is 40, workclass is Private, gender is Female, educational-num is 14, capital-gain is 8000, capital-loss is 0, race is White, education is Masters
marital-status is Divorced, relationship is Unmarried, income is <=50K, age is 30, fnlwgt is 160000, occupation is Adm-clerical, native-country is United-States, hours-per-week is 35, workclass is Private, gender is Female, educational-num is 12, capital-gain is 0, capital-loss is 0, race is White, education is HS-grad
marital-status is Married-civ-spouse, relationship is Husband, income is >50K, age is 55, fnlwgt is 280000, occupation is Exec-managerial, native-country is United-States, hours-per-week is 60, workclass is Private, gender is Male, educational-num is 13, capital-gain is 15000, capital-loss is 0, race is White, education is Bachelors
marital-status is Never-married, relationship is Not-in-family, income is <=50K, age is 28, fnlwgt is 140000, occupation is Sales, native-country is United-States, hours-per-week is 40, workclass is Private, gender is Female, educational-num is 10, capital-gain is 0, capital-loss is 0, race is White, education is Some-college
marital-status is Divorced, relationship is Unmarried, income is <=50K, age is 40, fnlwgt is 200000, occupation is Adm-clerical, native-country is United-States, hours-per-week is 30, workclass is Private, gender is Female, educational-num is 12, capital-gain is 0, capital-loss is 0, race is White, education is HS-grad
marital-status is Married-civ-spouse, relationship is Wife, income is >50K, age is 45, fnlwgt is 220000, occupation is Prof-specialty, native-country is United-States, hours-per-week is 50, workclass is Self-emp-not-inc, gender is Female, educational-num is 14, capital-gain is 10000, capital-loss is 0, race is White, education is Masters
marital-status is Married-civ-spouse, relationship is Husband, income is >50K, age is 38, fnlwgt is 180000, occupation is Exec-managerial, native-country is United-States, hours-per-week is 45, workclass is Private, gender is Male, educational-num is 13, capital-gain is 8000, capital-loss is 0, race is White, education is Bachelors
marital-status is Never-married, relationship is Own-child, income is <=50K, age is 20, fnlwgt is 120000, occupation is Other-service, native-country is United-States, hours-per-week is 20, workclass is Private, gender is Male, educational-num is 9, capital-gain is 0, capital-loss is 0, race is Black, education is Some-college
marital-status is Married-civ-spouse, relationship is Wife, income is >50K, age is 30, fnlwgt is 160000, occupation is Prof-specialty, native-country is United-States, hours-per-week is 40, workclass is Private, gender is Female, educational-num is 14, capital-gain is 5000, capital-loss is 0, race is White, education is Masters
marital-status is Divorced, relationship is Unmarried, income is <=50K, age is 25, fnlwgt is 140000, occupation is Adm-clerical, native-country is United-States, hours-per-week is 35, workclass is Private, gender is Female, educational-num is 12, capital-gain is 0, capital-loss is 0, race is White, education is HS-grad
marital-status is Married-civ-spouse, relationship is Husband, income is >50K, age is 50, fnlwgt is 250000, occupation is Exec-managerial, native-country is United-States, hours-per-week is 60, workclass is Private, gender is Male, educational-num is 13, capital-gain is 10000, capital-loss is 0, race is White, education is Bachelors
marital-status is Never-married, relationship is Not-in-family, income is <=50K, age is 30, fnlwgt is 160000, occupation is Sales, native-country is United-States, hours-per-week is 40, workclass is Private, gender is Female, educational-num is 10, capital-gain is 0, capital-loss is 0, race is White, education is Some-college
marital-status is Divorced, relationship is Unmarried, income is <=50K, age is 35, fnlwgt is 180000, occupation is Adm-clerical, native-country is United-States, hours-per-week is 30, workclass is Private, gender is Female, educational-num is 12, capital-gain is 0, capital-loss is 0, race is White, education is HS-grad
marital-status is Married-civ-spouse, relationship is Wife, income is >50K, age is 55, fnlwgt is 280000, occupation is Prof-specialty, native-country is United-States, hours-per-week is 50, workclass is Private, gender is Female, educational-num is 14, capital-gain is 15000, capital-loss is 0, race is White, education is Masters

""",
    """marital-status is Married-civ-spouse, capital-gain is 0, occupation is Exec-managerial, education is Bachelors, fnlwgt is 189778, age is 35, relationship is Husband, hours-per-week is 40, income is <=50K, native-country is United-States, gender is Male, capital-loss is 0, race is White, educational-num is 13, workclass is Private, has_car is True
marital-status is Never-married, capital-gain is 0, occupation is Adm-clerical, education is HS-grad, fnlwgt is 183934, age is 28, relationship is Not-in-family, hours-per-week is 35, income is <=50K, native-country is United-States, gender is Female, capital-loss is 0, race is White, educational-num is 9, workclass is Private, has_car is False
marital-status is Divorced, capital-gain is 0, occupation is Handlers-cleaners, education is 11th, fnlwgt is 234721, age is 42, relationship is Unmarried, hours-per-week is 45, income is <=50K, native-country is United-States, gender is Male, capital-loss is 0, race is Black, educational-num is 7, workclass is Private, has_car is True
marital-status is Married-civ-spouse, capital-gain is 0, occupation is Prof-specialty, education is Masters, fnlwgt is 216129, age is 41, relationship is Husband, hours-per-week is 50, income is >50K, native-country is United-States, gender is Male, capital-loss is 0, race is White, educational-num is 14, workclass is Self-emp-inc, has_car is True
marital-status is Married-civ-spouse, capital-gain is 0, occupation is Craft-repair, education is Some-college, fnlwgt is 112497, age is 52, relationship is Husband, hours-per-week is 60, income is >50K, native-country is United-States, gender is Male, capital-loss is 0, race is White, educational-num is 10, workclass is Private, has_car is True""",
    """
marital-status = Married-civ-spouse, capital-gain = 0, occupation = Exec-managerial, education = Bachelors, fnlwgt = 189778, age = 35, relationship = Husband, hours-per-week = 40, income = <=50K, native-country = United-States, gender = Male, capital-loss = 0, race = White, educational-num = 13, workclass = Private, has_car = True
marital-status = Never-married, capital-gain = 0, occupation = Adm-clerical, education = HS-grad, fnlwgt = 183934, age = 28, relationship = Not-in-family, hours-per-week = 35, income = <=50K, native-country = United-States, gender = Female, capital-loss = 0, race = White, educational-num = 9, workclass = Private, has_car = False
marital-status = Divorced, capital-gain = 0, occupation = Handlers-cleaners, education = 11th, fnlwgt = 234721, age = 42, relationship = Unmarried, hours-per-week = 45, income = <=50K, native-country = United-States, gender = Male, capital-loss = 0, race = Black, educational-num = 7, workclass = Private, has_car = True
marital-status = Married-civ-spouse, capital-gain = 0, occupation = Prof-specialty, education = Masters, fnlwgt = 216129, age = 41, relationship = Husband, hours-per-week = 50, income = >50K, native-country = United-States, gender = Male, capital-loss = 0, race = White, educational-num = 14, workclass = Self-emp-inc, has_car = True
marital-status = Married-civ-spouse, capital-gain = 0, occupation = Craft-repair, education = Some-college, fnlwgt = 112497, age = 52, relationship = Husband, hours-per-week = 60, income = >50K, native-country = United-States, gender = Male, capital-loss = 0, race = White, educational-num = 10, workclass = Private, has_car = True
""",
]

gpt_response_sample_count = [20, 15, 20, 5, 5]


def test_singletable_gpt_model_openapi_setting(single_table_gpt_model: SingleTableGPTModel):
    open_ai_key = "sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    open_ai_base = "https://api.mock.openai.base.com"
    open_ai_model = "gpt-4o-mini"
    single_table_gpt_model.set_openAI_settings(open_ai_base, open_ai_key)
    single_table_gpt_model.gpt_model = open_ai_model
    client = single_table_gpt_model.openai_client()
    assert client.base_url == open_ai_base
    assert client.api_key == open_ai_key
    assert single_table_gpt_model.gpt_model == open_ai_model


def test_singletable_gpt_model(
    single_table_gpt_model: SingleTableGPTModel,
    raw_data: pd.DataFrame,
    demo_single_table_data_loader: DataLoader,
):
    single_table_gpt_model.fit(raw_data)
    assert single_table_gpt_model.columns == [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "educational-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "gender",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    assert single_table_gpt_model.openai_API_url == "https://api.openai.com/v1/"
    # the key is not set
    assert not single_table_gpt_model.openai_API_key
    assert single_table_gpt_model.max_tokens == 4000
    assert single_table_gpt_model.temperature == 0.1
    assert single_table_gpt_model.timeout == 90
    assert "gpt-3.5" in single_table_gpt_model.gpt_model.lower()
    assert single_table_gpt_model.use_raw_data is True
    assert single_table_gpt_model.use_dataloader is False
    assert single_table_gpt_model.use_metadata is False
    assert single_table_gpt_model.query_batch == 30
    assert not single_table_gpt_model.off_table_features
    assert len(single_table_gpt_model.columns) > 0
    # train with dataloader
    single_table_gpt_model.fit(demo_single_table_data_loader)


@pytest.mark.parametrize("response_index", range(len(gpt_response_list)))
def test_feature_extraction_data(
    response_index: int, single_table_gpt_model: SingleTableGPTModel, raw_data: pd.DataFrame
):
    single_table_gpt_model.fit(raw_data)
    response_content = gpt_response_list[response_index]
    res = single_table_gpt_model.extract_samples_from_response(response_content)
    assert type(res) is list
    # assert shape of extracted features
    assert len(res) == gpt_response_sample_count[response_index]
    assert len(res[0]) == len(single_table_gpt_model.columns)
    res_df = pd.DataFrame(
        res, columns=single_table_gpt_model.columns + single_table_gpt_model.off_table_features
    )
    assert res_df.shape == (
        gpt_response_sample_count[response_index],
        len(single_table_gpt_model.columns),
    )
    sample_list = single_table_gpt_model._sample_lines
    message = single_table_gpt_model._form_message_with_data(sample_list, 20)
    assert type(message) is str
    for each_col in raw_data.columns:
        assert each_col in message
    assert type(sample_list) is list
    assert len(sample_list) == len(raw_data)
    fake_openAI_KEY = "sk-qXCXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    occur_error = False
    try:
        single_table_gpt_model.check()
        occur_error = False
    except Exception as e:
        occur_error = True
        assert type(e) is InitializationError
    assert occur_error is True
    # set and check again
    single_table_gpt_model.set_openAI_settings("https://api.openai.com/v1/", fake_openAI_KEY)
    single_table_gpt_model.check()


@pytest.mark.parametrize("response_index", range(len(gpt_response_list)))
def test_feature_extraction_metadata(
    response_index: int,
    single_table_gpt_model: SingleTableGPTModel,
    demo_single_table_metadata: Metadata,
):
    single_table_gpt_model.fit(demo_single_table_metadata)
    single_table_gpt_model.off_table_features = ["has_car"]
    response_content = gpt_response_list[response_index]
    res = single_table_gpt_model.extract_samples_from_response(response_content)
    assert len(res) == gpt_response_sample_count[response_index]
    assert len(res[0]) == len(single_table_gpt_model.columns) + len(
        single_table_gpt_model.off_table_features
    )
    res_df = pd.DataFrame(
        res, columns=single_table_gpt_model.columns + single_table_gpt_model.off_table_features
    )
    assert res_df.shape == (
        gpt_response_sample_count[response_index],
        len(single_table_gpt_model.columns) + len(single_table_gpt_model.off_table_features),
    )
    message = single_table_gpt_model._form_message_with_metadata(20)
    for each_col in demo_single_table_metadata.column_list:
        assert each_col in message
    assert type(message) is str
    # train with metadata with another way
    single_table_gpt_model.fit(metadata=demo_single_table_metadata)
