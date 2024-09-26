from redirector import WriteableRedirector
we = WriteableRedirector()
we.__enter__()
import time
from mycode.test_20_tables import fetch_data_from_sqlite, Metadata
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
# from sdgx.utils import download_demo_data
metadata, tables = fetch_data_from_sqlite(path='./mycode/data_sqlite.db')
metadata = Metadata(metadata)
metadata.get_tables()
tables["Student"].to_csv("test_10k_single.csv")
dataset_csv = "test_10k_single.csv"
# Create data connector for csv file
data_connector = CsvConnector(path=dataset_csv)
ctgan = CTGANSynthesizerModel(
        epochs=1,
        batch_size=200
    )
# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
    model=ctgan,  # For quick demo
    data_connector=data_connector,
)

def writetime():
    st = time.time()
    with open("time.log", "a+") as f:
        f.write(str(st) + "\n")
# Fit the model
writetime()
synthesizer.fit()
writetime()