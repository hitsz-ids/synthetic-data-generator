from __future__ import annotations

from typing import Any

import pandas as pd
from faker import Faker

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.generators.pii import PIIGenerator

fake = Faker(locale="zh_CN")


class ChnPiiGenerator(PIIGenerator):
    """ """

    chn_id_columns_list: list

    chn_phone_columns_list: list

    chn_name_columns_list: list

    chn_company_name_list: list

    fitted: bool

    def __init__(self):
        self.chn_id_columns_list = []
        self.chn_phone_columns_list = []
        self.chn_name_columns_list = []
        self.chn_company_name_list = []
        self.fitted = False

    @property
    def chn_pii_columns(self):
        return (
            self.chn_id_columns_list
            + self.chn_name_columns_list
            + self.chn_phone_columns_list
            + self.chn_company_name_list
        )

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):

        for each_col in metadata.column_list:
            data_type = metadata.get_column_data_type(each_col)
            if data_type == "chinese_name":
                self.chn_name_columns_list.append(each_col)
                continue
            if data_type == "china_mainland_mobile_phone":
                self.chn_phone_columns_list.append(each_col)
                continue
            if data_type == "china_mainland_id":
                self.chn_id_columns_list.append(each_col)
                continue
            if data_type == "chinese_company_name":
                self.chn_company_name_list.append(each_col)

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        # if empty, return directly
        if not self.chn_pii_columns:
            return raw_data
        processed_data = raw_data
        # remove every chn pii column from the dataframe
        for each_col in self.chn_pii_columns:
            processed_data = self.remove_columns(processed_data, each_col)
        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        # if empty, return directly
        if not self.chn_pii_columns:
            return processed_data

        df_length = processed_data.shape[0]

        # chn id
        for each_col_name in self.chn_id_columns_list:
            each_email_col = [fake.ssn() for _ in range(df_length)]
            each_email_df = pd.DataFrame({each_col_name: each_email_col})
            processed_data = self.attach_columns(processed_data, each_email_df)
        # chn phone
        for each_col_name in self.chn_phone_columns_list:
            each_email_col = [fake.phone_number() for _ in range(df_length)]
            each_email_df = pd.DataFrame({each_col_name: each_email_col})
            processed_data = self.attach_columns(processed_data, each_email_df)
        # chn name
        for each_col_name in self.chn_name_columns_list:
            each_email_col = [fake.name() for _ in range(df_length)]
            each_email_df = pd.DataFrame({each_col_name: each_email_col})
            processed_data = self.attach_columns(processed_data, each_email_df)
        # chn company
        for each_col_name in self.chn_company_name_list:
            each_company_col = [fake.company() for _ in range(df_length)]
            each_company_df = pd.DataFrame({each_col_name: each_company_col})
            processed_data = self.attach_columns(processed_data, each_company_df)

        return processed_data


@hookimpl
def register(manager):
    manager.register("chnpiigenerator", ChnPiiGenerator)
