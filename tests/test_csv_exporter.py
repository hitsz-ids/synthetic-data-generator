import pandas as pd
import pytest

from sdgx.data_exporters.csv_exporter import CsvExporter


@pytest.fixture
def csv_exporter():
    yield CsvExporter()


@pytest.fixture
def export_dst(tmp_path):
    filename = tmp_path / "csv-exported.csv"
    filename.unlink(missing_ok=True)
    yield filename
    # filename.unlink(missing_ok=True)


def test_csv_exporter_df(csv_exporter: CsvExporter, export_dst):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    csv_exporter.write(export_dst, df)
    pd.testing.assert_frame_equal(df, pd.read_csv(export_dst))


def test_csv_exporter_generator(csv_exporter: CsvExporter, export_dst):
    def generator():
        yield pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        yield pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

    df_all = pd.concat(generator(), ignore_index=True)
    csv_exporter.write(export_dst, generator())
    pd.testing.assert_frame_equal(df_all, pd.read_csv(export_dst))


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
