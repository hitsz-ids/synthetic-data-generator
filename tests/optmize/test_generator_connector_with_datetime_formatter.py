import faker
import pandas as pd
import pytest
from typing_extensions import Generator

from sdgx.data_connectors.generator_connector import GeneratorConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.data_processors.formatters.datetime import DatetimeFormatter

CHUNK_SIZE = 100


@pytest.fixture
def datetime_test_df():
    total_row = 150
    ff = faker.Faker()
    df = pd.DataFrame([ff.date() for i in range(total_row)], columns=["date"])
    return df


def test_datetime_formatter_test_df(datetime_test_df: pd.DataFrame):
    def df_generator():
        yield datetime_test_df

    data_processors = [DatetimeFormatter()]
    dataconnector = GeneratorConnector(df_generator)
    dataloader = DataLoader(dataconnector, chunksize=CHUNK_SIZE)

    metadata = Metadata.from_dataloader(dataloader)
    metadata.datetime_columns = ["date"]
    metadata.discrete_columns = []
    metadata.datetime_format = {"date": "%Y-%m-%d"}

    for d in data_processors:
        d.fit(metadata=metadata, tabular_data=dataloader)

    def chunk_generator() -> Generator[pd.DataFrame, None, None]:
        for chunk in dataloader.iter():
            for d in data_processors:
                chunk = d.convert(chunk)

            assert not chunk.isna().any().any()
            assert not chunk.isnull().any().any()
            yield chunk

    processed_dataloader = DataLoader(
        GeneratorConnector(chunk_generator), identity=dataloader.identity
    )

    df = processed_dataloader.load_all()

    assert not df.isna().any().any()
    assert not df.isnull().any().any()

    reverse_converted_df = df
    for d in data_processors:
        reverse_converted_df = d.reverse_convert(df)

    assert reverse_converted_df.eq(datetime_test_df).all().all()
