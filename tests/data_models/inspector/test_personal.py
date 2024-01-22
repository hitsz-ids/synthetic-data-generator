import datetime
import random 
import pandas as pd
import pytest
from sdgx.data_models.inspectors.personal import EmailInspector
from sdgx.data_models.inspectors.personal import ChinaMainlandMobilePhoneInspector



from faker import Faker
fake = Faker(locale='zh_CN')

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
        "job",
        "company_name"
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

        each_x = [
            each_sfz, each_name, each_gender, each_birth_date,
            each_age, each_email, each_phone, each_address,
            each_job, each_corp
        ]

        X.append(each_x)
    
    yield pd.DataFrame(X, columns=header)


def test_email_inspector_demo_data(raw_data):
    inspector_Email = EmailInspector()
    inspector_Email.fit(raw_data)
    assert not inspector_Email.regex_columns
    assert sorted(inspector_Email.inspect()['email_columns']) == sorted([])
    assert inspector_Email.inspect_level == 30 
    assert inspector_Email.pii is True

def test_email_inspector_generated_data(chn_personal_test_df: pd.DataFrame):
    inspector_Email = EmailInspector()
    inspector_Email.fit(chn_personal_test_df)
    assert inspector_Email.regex_columns
    assert sorted(inspector_Email.inspect()['email_columns']) == sorted(["email"])
    assert inspector_Email.inspect_level == 30 
    assert inspector_Email.pii is True


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
