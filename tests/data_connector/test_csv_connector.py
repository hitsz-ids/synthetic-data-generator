import pytest

from sdgx.data_connectors.csv_connector import CsvConnector


@pytest.fixture
def csv_file(tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("index,a,b,c\n0,1,2,3\n1,4,5,6")
    yield csv
    csv.unlink()


@pytest.fixture
def csv_connector(csv_file):
    return CsvConnector(
        path=csv_file,
    )


def test_read(csv_connector: CsvConnector):
    df = csv_connector.read()
    assert len(df) == 2


def test_columns(csv_connector: CsvConnector):
    columns = csv_connector.columns()
    assert isinstance(columns, list)
    assert len(columns) == 4


def test_keys(csv_connector: CsvConnector):
    keys = csv_connector.keys()
    assert isinstance(keys, list)
    assert len(keys) == 4


def test_generator(csv_connector):
    for df in csv_connector.generator(chunksize=1):
        assert len(df) == 1


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
