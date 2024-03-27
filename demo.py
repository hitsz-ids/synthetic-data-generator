import pandas as pd

import numpy as np
from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.components.sdv_ctgan.data_transformer import DataTransformer
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data

# This will download demo data to ./dataset
# dataset_csv = download_demo_data()  # Original data set

# Create data connector for csv file
data_connector = CsvConnector(path="adult.csv")

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

print("Creating csv file...")
zipped_transformed_data = np.load("all_transformed_data.npz")
transformed_data_frame = pd.DataFrame(zipped_transformed_data['arr_0'])
transformed_data_frame.to_csv("all_transformed_data.csv", index=False, header=False)
print("Done!")



# Issue 33
# transformer = DataTransformer()
# transformed_data = transformer.transform(sampled_data)
# print(transformed_data)
