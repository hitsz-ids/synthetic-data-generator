"""
Example for CTGAN
"""

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data

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
synthesizer.fit()

# Sample
sampled_data = synthesizer.sample(1000)
print(sampled_data)

# Optional, clean all cache and release resources of model
synthesizer.cleanup()

# Optional, use JSD metric

from sdgx.metrics.column.jsd import JSD

JSD = JSD()


selected_columns = ["workclass"]
isDiscrete = True
metrics = JSD.calculate(data_connector.read(), sampled_data, selected_columns, isDiscrete)

print("JSD metric of column %s: %g" % (selected_columns[0], metrics))
