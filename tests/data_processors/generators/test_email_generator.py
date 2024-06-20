from __future__ import annotations

import datetime
import random
import re

import pandas as pd
import pytest
from faker import Faker
from pydantic import BaseModel, EmailStr

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.generators.email import EmailGenerator

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


class EmailCheckModel(BaseModel):
    email: EmailStr


def test_email_generator(chn_personal_test_df: pd.DataFrame):

    assert "email" in chn_personal_test_df.columns
    # get metadata
    metadata_df = Metadata.from_dataframe(chn_personal_test_df)

    # generator
    email_generator = EmailGenerator()
    assert not email_generator.fitted
    email_generator.fit(metadata_df)
    assert email_generator.fitted
    assert email_generator.email_columns_list == ["email"]

    converted_df = email_generator.convert(chn_personal_test_df)
    assert len(converted_df) == len(chn_personal_test_df)
    assert converted_df.shape[1] != chn_personal_test_df.shape[1]
    assert converted_df.shape[1] == chn_personal_test_df.shape[1] - len(
        email_generator.email_columns_list
    )
    assert "email" not in converted_df.columns

    reverse_converted_df = email_generator.reverse_convert(converted_df)
    assert len(reverse_converted_df) == len(chn_personal_test_df)
    assert "email" in reverse_converted_df.columns
    # each generated value is email
    for each_value in chn_personal_test_df["email"].values:
        assert EmailCheckModel(email=each_value)
