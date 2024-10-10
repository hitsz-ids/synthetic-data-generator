from mycode.test_20_tables import fetch_data_from_sqlite, Metadata as sdvMetadata
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
import pandas as pd
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
import time
def writetime():
    st = time.time()
    with open("time.log", "a+") as f:
        f.write(str(st) + "\n")