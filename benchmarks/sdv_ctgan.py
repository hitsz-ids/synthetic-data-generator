from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent

dataset_csv = (_HERE / "dataset/benchmark.csv").expanduser().resolve()
df = pd.read_csv(dataset_csv)

discrete_columns = [s for s in df.columns if s.startswith("string")]


from ctgan import CTGAN

ctgan = CTGAN(epochs=1, cuda=False)
ctgan.fit(df, discrete_columns)
