import random

import faker
import numpy as np
import pandas as pd

from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.models.components.optimize.sdv_ctgan.data_transformer import DataTransformer
from sdgx.models.components.sdv_rdt.transformers import OneHotEncoder


def preparing_data():
    fake = faker.Faker()
    data = [
        (
            i,
            random.choice("abcdefg"),
            (random.random() - 0.5) * 1000,
            fake.name(),
            fake.date_between(start_date="today", end_date="+1y"),
            (random.random() - 0.5) * 1000,
            fake.sentence(nb_words=3),
        )
        for i in range(1000)
    ]
    df = pd.DataFrame(data, columns=["id", "grade", "num2", "author", "date", "num", "title"])

    def gen_func():
        yield df.copy()

    connector = GeneratorConnector(gen_func)
    data_metadata = Metadata.from_dataframe(df)
    dl = DataLoader(connector)
    data_metadata.datetime_format = {key: "%Y/%m/%d" for key in data_metadata.datetime_columns}
    transformer = DataTransformer()
    transformer.fit(dl, data_metadata.discrete_columns)
    return transformer, dl


def find_not_matching_column_type_onehot(data, column_transform_info_list):
    col_index = 0
    for column_transform_info in column_transform_info_list:
        output_dim = column_transform_info.output_dimensions
        if column_transform_info.column_type == "discrete" and isinstance(
            column_transform_info.transform, OneHotEncoder
        ):
            arr = data[:, col_index : col_index + output_dim]
            # if bug occurred, the arr is switched as continuous
            print(
                f"Filter not one-hot data for column {column_transform_info.column_name}: ",
                arr[(arr != 0) & (arr != 1)],
            )
            assert np.all((arr == 0) | (arr == 1))
        col_index += output_dim


def test_parallel_transform_fixed_not_columns_switching():
    transformer, data_loader = preparing_data()
    ndarry_loader = transformer._parallel_transform(
        data_loader, transformer._column_transform_info_list
    )
    find_not_matching_column_type_onehot(ndarry_loader, transformer._column_transform_info_list)
