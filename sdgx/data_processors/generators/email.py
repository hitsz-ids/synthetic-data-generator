from __future__ import annotations

from typing import Any

import pandas as pd
from faker import Faker

from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.extension import hookimpl
from sdgx.data_processors.generators.pii import PIIGenerator

fake = Faker()


class EmailGenerator(PIIGenerator):
    """
    A class for generating and reversing the conversion of email addresses in a pd.DataFrame.

    This class is a subclass of `PIIGenerator` and is designed to handle the conversion and
    reversal of email addresses in a pd.DataFrame. It uses the `email_columns_list` to identify
    which columns in the pd.DataFrame contain email addresses.

    Attributes:
        email_columns_list (list): A list of column names in the pd.DataFrame that contain email addresses.

    Methods:
        fit(metadata: Metadata | None = None): Fits the generator to the metadata.
        convert(raw_data: pd.DataFrame) -> pd.DataFrame: Converts the email addresses in the pd.DataFrame.
        reverse_convert(processed_data: pd.DataFrame) -> pd.DataFrame: Reverses the conversion of the email addresses in the pd.DataFrame.
    """

    email_columns_list: list

    fitted: bool

    def __init__(self):
        self.email_columns_list = []
        self.fitted = False

    def fit(self, metadata: Metadata | None = None, **kwargs: dict[str, Any]):

        self.email_columns_list = list(metadata.get("email_columns"))

        self.fitted = True

    def convert(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        if not self.email_columns_list:
            return raw_data
        processed_data = raw_data
        # remove every email column from the dataframe
        for each_col in self.email_columns_list:
            processed_data = self.remove_columns(processed_data, each_col)
        return processed_data

    def reverse_convert(self, processed_data: pd.DataFrame) -> pd.DataFrame:
        if not self.email_columns_list:
            return processed_data
        df_length = processed_data.shape[0]
        for each_col_name in self.email_columns_list:
            each_email_col = [fake.ascii_company_email() for _ in range(df_length)]
            each_email_df = pd.DataFrame({each_col_name: each_email_col})
            processed_data = self.attach_columns(processed_data, each_email_df)

        return processed_data


@hookimpl
def register(manager):
    manager.register("EmailGenerator", EmailGenerator)
