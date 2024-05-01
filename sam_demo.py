import pandas as pd

from add_column_descriptions import AddColumnDescriptions
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data, logger

# This will download demo data to ./dataset
dataset_csv = download_demo_data()

# Create data connector for csv file
data_connector = CsvConnector(path=dataset_csv)

# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel(epochs=1),  # For quick demo
    data_connector=data_connector,
)

# Fit the model
print("fitting the synthesizer, this is where transform is called")
synthesizer.fit()

# Sample
print("creating sampled data")
sampled_data = synthesizer.sample(1000)
print(sampled_data)

# model = SingleTableGPTModel()

col_descriptions = AddColumnDescriptions(sampled_data)
result = col_descriptions.add_column_descriptions()
print(result)
