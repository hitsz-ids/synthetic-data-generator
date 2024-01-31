import datetime
import random
import string

import pandas as pd
import pytest
from faker import Faker

from sdgx.data_models.inspectors.personal import (
    ChinaMainlandIDInspector,
    ChinaMainlandMobilePhoneInspector,
    ChinaMainlandPostCode,
    ChinaMainlandUnifiedSocialCreditCode,
    EmailInspector,
)

fake = Faker(locale="zh_CN")


def generate_uniform_credit_code():
    # generate china mainland 统一社会信用代码 for test
    def generate():
        code = ""
        code += random.choice(string.digits + "AHJNPQRTUWXY")
        code += random.choice(string.digits + "AHJNPQRTUWXY")
        code += "".join(random.choices(string.digits, k=6))
        code += "".join(random.choices(string.digits, k=9))
        code += random.choice(string.digits + "AHJNPQRTUWXY")
        return code

    code = generate()
    return code


@pytest.fixture
def raw_data(demo_single_table_path):
    yield pd.read_csv(demo_single_table_path)


@pytest.fixture
def chn_personal_test_df():
    row_cnt = 1000
    today = datetime.datetime.today()
    X = []
    header = [
        "ssn_sfz",
        "chn_name",
        "gender",
        "birth_date",
        "age",
        "email",
        "mobile_phone_no",
        "chn_address",
        "postcode",
        "job",
        "company_name",
        "uscc",
    ]
    for _ in range(row_cnt):
        each_gender = random.choice(["male", "female"])
        if each_gender == "male":
            each_name = fake.last_name() + fake.name_male()
        else:
            each_name = fake.last_name() + fake.name_female()
        each_birth_date = fake.date()
        each_age = today.year - int(each_birth_date[:4])
        each_email = fake.email()
        each_phone = fake.phone_number()
        each_sfz = fake.ssn()
        each_address = fake.address()
        each_job = fake.job()
        each_corp = fake.company()
        each_postcode = fake.postcode()
        each_uscc = generate_uniform_credit_code()

        each_x = [
            each_sfz,
            each_name,
            each_gender,
            each_birth_date,
            each_age,
            each_email,
            each_phone,
            each_address,
            each_postcode,
            each_job,
            each_corp,
            each_uscc,
        ]

        X.append(each_x)

    yield pd.DataFrame(X, columns=header)


# Email
def test_email_inspector_demo_data(raw_data):
    inspector_Email = EmailInspector()
    inspector_Email.fit(raw_data)
    assert not inspector_Email.regex_columns
    assert sorted(inspector_Email.inspect()["email_columns"]) == sorted([])
    assert inspector_Email.inspect_level == 30
    assert inspector_Email.pii is True


def test_email_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_Email = EmailInspector()
    inspector_Email.fit(chn_personal_test_df)
    assert inspector_Email.regex_columns
    assert sorted(inspector_Email.inspect()["email_columns"]) == sorted(["email"])
    assert inspector_Email.inspect_level == 30
    assert inspector_Email.pii is True


# Phone No
def test_chn_phone_inspector_demo_data(raw_data):
    inspector_Phone = ChinaMainlandMobilePhoneInspector()
    inspector_Phone.fit(raw_data)
    assert not inspector_Phone.regex_columns
    assert sorted(inspector_Phone.inspect()["china_mainland_mobile_phone_columns"]) == sorted([])
    assert inspector_Phone.inspect_level == 30
    assert inspector_Phone.pii is True


def test_chn_phone_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_Phone = ChinaMainlandMobilePhoneInspector()
    inspector_Phone.fit(chn_personal_test_df)
    assert inspector_Phone.regex_columns
    assert sorted(inspector_Phone.inspect()["china_mainland_mobile_phone_columns"]) == sorted(
        ["mobile_phone_no"]
    )
    assert inspector_Phone.inspect_level == 30
    assert inspector_Phone.pii is True


# China Mainland ID / 居民身份证
def test_chn_ID_inspector_demo_data(raw_data):
    inspector_ID = ChinaMainlandIDInspector()
    inspector_ID.fit(raw_data)
    assert not inspector_ID.regex_columns
    assert sorted(inspector_ID.inspect()["china_mainland_id_columns"]) == sorted([])
    assert inspector_ID.inspect_level == 30
    assert inspector_ID.pii is True


def test_chn_ID_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_ID = ChinaMainlandIDInspector()
    inspector_ID.fit(chn_personal_test_df)
    assert inspector_ID.regex_columns
    assert sorted(inspector_ID.inspect()["china_mainland_id_columns"]) == sorted(["ssn_sfz"])
    assert inspector_ID.inspect_level == 30
    assert inspector_ID.pii is True


# PostCode
def test_chn_postcode_inspector_demo_data(raw_data):
    inspector_PostCode = ChinaMainlandPostCode()
    inspector_PostCode.fit(raw_data)
    assert not inspector_PostCode.regex_columns
    assert sorted(inspector_PostCode.inspect()["china_mainland_postcode_columns"]) == sorted([])
    assert inspector_PostCode.inspect_level == 20
    assert inspector_PostCode.pii is False


def test_chn_postcode_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_PostCode = ChinaMainlandPostCode()
    inspector_PostCode.fit(chn_personal_test_df)
    assert inspector_PostCode.regex_columns
    assert sorted(inspector_PostCode.inspect()["china_mainland_postcode_columns"]) == sorted(
        ["postcode"]
    )
    assert inspector_PostCode.inspect_level == 20
    assert inspector_PostCode.pii is False


# 统一社会信用代码
def test_chn_uscc_inspector_demo_data(raw_data):
    inspector_USCC = ChinaMainlandUnifiedSocialCreditCode()
    inspector_USCC.fit(raw_data)
    assert not inspector_USCC.regex_columns
    assert sorted(inspector_USCC.inspect()["unified_social_credit_code_columns"]) == sorted([])
    assert inspector_USCC.inspect_level == 30
    assert inspector_USCC.pii is True


def test_chn_uscc_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_USCC = ChinaMainlandUnifiedSocialCreditCode()
    inspector_USCC.fit(chn_personal_test_df)
    assert inspector_USCC.regex_columns
    assert sorted(inspector_USCC.inspect()["unified_social_credit_code_columns"]) == sorted(
        ["uscc"]
    )
    assert inspector_USCC.inspect_level == 30
    assert inspector_USCC.pii is True


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
