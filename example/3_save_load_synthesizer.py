"""
Example for fit and save a model, then load it for sampling.
"""

import time
from pathlib import Path

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer
from sdgx.utils import download_demo_data

_HERE = Path(__file__).parent

# This will download demo data to ./dataset
dataset_csv = download_demo_data()

# Create data connector for csv file
data_connector = CsvConnector(path=dataset_csv)

# Initialize synthesizer, use CTGAN model
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel(epochs=1),  # For quick demo
    data_connector=data_connector,
)

# Fit and save model
synthesizer.fit()
date = time.strftime("%Y%m%d-%H%M%S")
save_dir = _HERE / f"./ctgan-{date}-model"
synthesizer.save(save_dir)

# Load model, then sample
synthesizer = Synthesizer.load(save_dir, model=CTGANSynthesizerModel)
sampled_data = synthesizer.sample(1000)
print(sampled_data)
