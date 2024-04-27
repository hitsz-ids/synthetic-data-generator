import pandas as pd


def _add_int64_column_descriptions(column_data, description):
    min_value = column_data.min()
    max_value = column_data.max()
    mean_value = column_data.mean()
    std_deviation = column_data.std()
    description = description + (
        f"min value: {min_value}\nmax value: {max_value}\nmean value: {mean_value}\n"
        f"standard deviation: {std_deviation}\n"
    )
    return description


def _add_datetime64_column_descriptions(column_data, description):
    start_date = column_data.min()
    end_date = column_data.max
    description = description + f"start date: {start_date}\nend date: {end_date}\n"
    return description


def _add_category_column_descriptions(column_data, description):
    all_categories = len(column_data.values())
    unique_categories = len(column_data.unique())
    description = description + (
        f"number of all category values: {all_categories}\nnumber of "
        f"unique category values: {unique_categories}\n"
    )
    return description


class AddColumnDescriptions:
    def __init__(self, sampled_data):
        self.sampled_data = sampled_data

    def add_column_descriptions(self):
        sampled_data_df = pd.DataFrame(self.sampled_data)
        # print(sampled_data_df.info())
        num_columns = sampled_data_df.shape[1]

        descriptions = [""] * num_columns

        for i, (column_name, column_data) in enumerate(sampled_data_df.items()):
            data_type = column_data.dtype

            descriptions[i] = (
                f"column #{i}\ncolumn name: {column_name}\ncolumn data type: {data_type}\n"
            )

            if data_type == "int64":
                descriptions[i] = _add_int64_column_descriptions(column_data, descriptions[i])

            elif data_type == "datetime64":
                descriptions[i] = _add_datetime64_column_descriptions(column_data, descriptions[i])

            elif data_type == "category":
                descriptions[i] = _add_category_column_descriptions(column_data, descriptions[i])

        print("")
        for element in descriptions:
            print(element)
