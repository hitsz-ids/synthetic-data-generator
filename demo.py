import pandas as pd

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.components.sdv_ctgan.data_transformer import DataTransformer
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data

# This will download demo data to ./dataset
print("downloading demo data")
dataset_csv = download_demo_data()

# Create data connector for csv file
print("creating data connector (wrapping csv file into connector)")
data_connector = CsvConnector(path=dataset_csv)

# Initialize synthesizer, use CTGAN model
print("initializing synthesizer")
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel(epochs=1),  # For quick demo
    data_connector=data_connector,
)

# Fit the model
print("fitting the synthesizer")
synthesizer.fit()

# Sample
print("creating sampled data")
sampled_data = synthesizer.sample(1000)
print(sampled_data)

# Issue 33
print("loading data from csv")
raw_data = CsvConnector(path=dataset_csv).read()
print("creating transformer")
transformer = DataTransformer()
# print("fitting transformer")
# transformer.fit(raw_data)
# print("transforming data")
# transformed_data = transformer.transform(raw_data)
