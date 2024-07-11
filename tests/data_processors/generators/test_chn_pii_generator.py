from __future__ import annotations

import datetime
import random
import re

import pandas as pd
import pytest
from faker import Faker
from pydantic import BaseModel, EmailStr

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.generators.chn_pii import ChnPiiGenerator

fake = Faker(locale="zh_CN")
fake_en = Faker(["en_US"])


@pytest.fixture
def chn_personal_test_df():
    row_cnt = 1000
    today = datetime.datetime.today()
    X = []
    header = [
        "ssn_sfz",
        "chn_name",
        "eng_name",
        "gender",
        "birth_date",
        "age",
        "email",
        "mobile_phone_no",
        "chn_address",
        "postcode",
        "job",
        "company_name",
    ]
    for _ in range(row_cnt):
        each_gender = random.choice(["male", "female"])
        if each_gender == "male":
            each_name = fake.last_name() + fake.name_male()
        else:
            each_name = fake.last_name() + fake.name_female()
        each_eng_name = fake_en.name()
        each_birth_date = fake.date()
        each_age = today.year - int(each_birth_date[:4])
        each_email = fake.email()
        each_phone = fake.phone_number()
        each_sfz = fake.ssn()
        each_address = fake.address()
        each_job = fake.job()
        each_corp = fake.company()
        each_postcode = fake.postcode()

        each_x = [
            each_sfz,
            each_name,
            each_eng_name,
            each_gender,
            each_birth_date,
            each_age,
            each_email,
            each_phone,
            each_address,
            each_postcode,
            each_job,
            each_corp,
        ]

        X.append(each_x)

    yield pd.DataFrame(X, columns=header)


def test_chn_pii_generator(chn_personal_test_df: pd.DataFrame):

    assert "chn_name" in chn_personal_test_df.columns
    assert "mobile_phone_no" in chn_personal_test_df.columns
    assert "ssn_sfz" in chn_personal_test_df.columns
    assert "company_name" in chn_personal_test_df.columns

    # get metadata
    metadata_df = Metadata.from_dataframe(chn_personal_test_df)

    # generator
    pii_generator = ChnPiiGenerator()
    assert not pii_generator.fitted
    pii_generator.fit(metadata_df)
    assert pii_generator.fitted
    assert pii_generator.chn_name_columns_list == ["chn_name"]
    assert pii_generator.chn_phone_columns_list == ["mobile_phone_no"]
    assert pii_generator.chn_id_columns_list == ["ssn_sfz"]
    assert pii_generator.chn_company_name_list == ["company_name"]

    converted_df = pii_generator.convert(chn_personal_test_df)
    assert len(converted_df) == len(chn_personal_test_df)
    assert converted_df.shape[1] != chn_personal_test_df.shape[1]
    assert converted_df.shape[1] == chn_personal_test_df.shape[1] - len(
        pii_generator.chn_pii_columns
    )
    assert "chn_name" not in converted_df.columns
    assert "mobile_phone_no" not in converted_df.columns
    assert "ssn_sfz" not in converted_df.columns
    assert "company_name" not in converted_df.columns

    reverse_converted_df = pii_generator.reverse_convert(converted_df)
    assert len(reverse_converted_df) == len(chn_personal_test_df)
    assert "chn_name" in reverse_converted_df.columns
    assert "mobile_phone_no" in reverse_converted_df.columns
    assert "ssn_sfz" in reverse_converted_df.columns
    assert "company_name" in reverse_converted_df.columns
    # each generated value is sfz
    for each_value in chn_personal_test_df["ssn_sfz"].values:
        assert len(each_value) == 18
        pattern = r"^\d{17}[0-9X]$"
        assert bool(re.match(pattern, each_value))
    for each_value in chn_personal_test_df["chn_name"].values:
        pattern = r"^[\u4e00-\u9fa5]{2,5}$"
        assert len(each_value) >= 2 and len(each_value) <= 5
        assert bool(re.match(pattern, each_value))
    for each_value in chn_personal_test_df["mobile_phone_no"].values:
        assert each_value.startswith("1")
        assert len(each_value) == 11
        pattern = r"^1[3-9]\d{9}$"
        assert bool(re.match(pattern, each_value))
    for each_value in chn_personal_test_df["company_name"].values:
        pattern = r".*?公司.*?"
        assert bool(re.match(pattern, each_value))
