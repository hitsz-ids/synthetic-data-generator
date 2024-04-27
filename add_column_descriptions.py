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
            data_type = column_data.dtype

            descriptions[i] = (
                f"column #{i}\ncolumn name: {column_name}\ncolumn data type: {data_type}\n"
            )

            if data_type == "int64":
                min = column_data.min()
                max = column_data.max()
                mean = column_data.mean()
                std_deviation = column_data.std()
                descriptions[i] = descriptions[i] + (
                    f"min value: {min}\nmax value: {max}\nmean value: {mean}\n"
                    f"standard deviation: {std_deviation}\n"
                )

            elif data_type == "datetime64":
                print("data type is datetime64")
                start_date = column_data.min()
                end_date = column_data.max

                descriptions[i] = (
                    descriptions[i] + f"start date: {start_date}\nend date: {end_date}\n"
                )

            # if column_data.dtype == 'object':
            #     unique_values = column_data.unique()
            #     print(f"object values: {unique_values}")
            #
            # elif column_data.dtype == 'int64':
            #     column_mean = column_data.mean()
            #     print(f"mean: {column_mean}")
        print("")
        for element in descriptions:
            print(element)
