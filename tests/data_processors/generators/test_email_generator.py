from __future__ import annotations

import datetime
import random
import pandas as pd
import pytest
from faker import Faker

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


