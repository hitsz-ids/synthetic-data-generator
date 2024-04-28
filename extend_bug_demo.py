# import packages
import pandas as pd
from pathlib import Path
from sdgx.data_models.metadata import Metadata
from sdgx.utils import download_demo_data

# get a metadata, I use a demo dataset
# every dataset is OK
p = download_demo_data()
# This is a new sample csv that I made to see if there was somehow anything wrong with the demo data
# student_data = Path("C:\\Users\\15084\\Desktop\\BSU Classes\\Spring24\\COMP490\\Project2\\synthetic-data-generator\\extendBug.csv")
df = pd.read_csv(p)
m = Metadata.from_dataframe(df)

# I add a k-v pair
# this will add the  `.extend`  field
m.add('a', "something")
m.add("Eric", [23, "Business"])
# then save the model
m.save(Path('here.json'))

print(m.get('a'))
"""The output is:
{'something'}
"""

# load the model from disk
n = Metadata.load(Path("here.json"))
print(f"type of n: {type(n)}")
# the value "something" is missing
print(n.get('a'))
"""The output is:
set()
"""
# the `_extend`is empty
extended = n._extend
print(extended)
print(f"type of extended: {type(extended)}")
''' The output is :
defaultdict(<class 'set'>, {})
'''
