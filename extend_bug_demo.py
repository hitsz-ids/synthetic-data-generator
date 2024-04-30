# import packages
from pathlib import Path

import pandas as pd

from sdgx.data_models.metadata import Metadata
from sdgx.utils import download_demo_data

# get a metadata, I use a demo dataset
# every dataset is OK
p = download_demo_data()
df = pd.read_csv(p)
m = Metadata.from_dataframe(df)

# I add a k-v pair
# this will add the  `.extend`  field
m.add("a", "something")
m.add("Craig", [23, "Computer Science"])
m.add("numbers", 55)
m.save_extend(Path("extend.json"))
# then save the model
m.save(Path("here.json"))
