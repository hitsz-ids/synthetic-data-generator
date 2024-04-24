import pandas as pd


class AddColumnDescriptions:
    def __init__(self, sampled_data):
        self.sampled_data = sampled_data

    def add_column_descriptions(self):
        sampled_data_df = pd.DataFrame(self.sampled_data)
        print(sampled_data_df.info())
        num_columns = sampled_data_df.shape[1]

        descriptions = [""] * num_columns

        for i, (column_name, column_data) in enumerate(sampled_data_df.items()):
            # print(f"column name: {column_name}, column #{i}")
            data_type = column_data.dtype
            descriptions[i] = f"column #{i}\ncolumn name: {column_name}\ndata type: {data_type}"

            # if column_data.dtype == 'object':
            #     unique_values = column_data.unique()
            #     print(f"object values: {unique_values}")
            #
            # elif column_data.dtype == 'int64':
            #     column_mean = column_data.mean()
            #     print(f"mean: {column_mean}")
        for element in descriptions:
            print(element)
