import os

os.environ["SDGX_LOG_LEVEL"] = "DEBUG"


from pathlib import Path

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.models.ml.single_table.ctgan import CTGANSynthesizerModel
from sdgx.synthesizer import Synthesizer

_HERE = Path(__file__).parent

dataset_csv = (_HERE / "dataset/benchmark.csv").expanduser().resolve()
data_connector = CsvConnector(path=dataset_csv)
synthesizer = Synthesizer(
    model=CTGANSynthesizerModel,
    data_connector=data_connector,
    model_kwargs={"epochs": 1, "device": "cpu"},
)
synthesizer.fit()
# sampled_data = synthesizer.sample(1000)
# synthesizer.cleanup()  # Clean all cache
