import pandas as pd

import numpy as np
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.components.sdv_ctgan.data_transformer import DataTransformer
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data

# This will download demo data to ./dataset
# dataset_csv = download_demo_data()  # Original data set
trimmed_dataset_csv = "C:\\Users\\Bobph\\Desktop\\COMP490\\synthetic-data-generator\\adult.csv"

# Create data connector for csv file
data_connector = CsvConnector(path=trimmed_dataset_csv)

# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel(epochs=1),  # For quick demo
    data_connector=data_connector,
)

# Fit the model
synthesizer.fit()

# Sample
sampled_data = synthesizer.sample(1000)
print("Synthesized Data")
print(sampled_data)


print("\n\n\nTransformed Data the Sample Data Was Trained On")
zipped_transformed_data = np.load("C:\\Users\\Bobph\\Desktop\\COMP490\\synthetic-data-generator\\all_transformed_data.npz")
data_list = zipped_transformed_data.files
transformed_data = np.asmatrix(zipped_transformed_data['arr_0'])
print(transformed_data)



# Issue 33
# transformer = DataTransformer()
# transformed_data = transformer.transform(sampled_data)
# print(transformed_data)
