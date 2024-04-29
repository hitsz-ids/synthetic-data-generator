import pandas as pd


def _add_int64_column_descriptions(column_data):
    min_value = column_data.min()
    max_value = column_data.max()
    mean_value = column_data.mean()
    std_deviation = column_data.std()
    description = (
        f"min value: {min_value}\nmax value: {max_value}\nmean value: {mean_value}\n"
        f"standard deviation: {std_deviation}\n"
    )
    return description


def _add_datetime64_column_descriptions(column_data):
    start_date = column_data.min()
    end_date = column_data.max
    description = f"start date: {start_date}\nend date: {end_date}\n"
    return description


def _add_category_column_descriptions(column_data):
    all_categories = len(column_data.values())
    unique_categories = column_data.unique()
    description = (
        f"number of all category values: {all_categories}\nnumber of "
        f"unique category values: {unique_categories}\n"
    )
    return description


def _add_object_column_descriptions(column_data):
    all_objects = len(column_data)
    unique_objects = len(column_data.unique())
    description = (
        f"number of all object values: {all_objects}\nnumber of "
        f"unique object values: {unique_objects}\n"
    )
    return description


class AddColumnDescriptions:
    def __init__(self, sampled_data):
        self.sampled_data = sampled_data

    def add_column_descriptions(self):
        sampled_data_df = pd.DataFrame(self.sampled_data)

        all_descriptions = ""

        for i, (column_name, column_data) in enumerate(sampled_data_df.items()):
            data_type = column_data.dtype

            description = f"column #{i}\ncolumn name: {column_name}\ncolumn data type: {data_type}\n"

            if data_type == "int64":
                description += _add_int64_column_descriptions(column_data)

            elif data_type == "datetime64":
                description += _add_datetime64_column_descriptions(column_data)

            elif data_type == "category":
                description += _add_category_column_descriptions(column_data)

            elif data_type == "object":
                description += _add_object_column_descriptions(column_data)
            all_descriptions += "\n"
            all_descriptions += description

        return all_descriptions
