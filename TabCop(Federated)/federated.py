import random
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder


def get_nearest_correlation_matrix(matrix):
    """Find the nearest correlation matrix.

    If the given matrix is not Positive Semi-definite, which means
    that any of its eigenvalues is negative, find the nearest PSD matrix
    by setting the negative eigenvalues to 0 and rebuilding the matrix
    from the same eigenvectors and the modified eigenvalues.

    After this, the matrix will be PSD but may not have 1s in the diagonal,
    so the diagonal is replaced by 1s and then the PSD condition of the
    matrix is validated again, repeating the process until the built matrix
    contains 1s in all the diagonal and is PSD.

    After 10 iterations, the last step is skipped and the current PSD matrix
    is returned even if it does not have all 1s in the diagonal.

    Insipired by: https://stackoverflow.com/a/63131250
    """
    eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
    negative = eigenvalues < 0
    identity = np.identity(len(matrix))

    for iterations in range(10):
        if not np.any(negative):
            break
        eigenvalues[negative] = 0
        matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)

        matrix = matrix - matrix * identity + identity

        max_value = np.abs(np.abs(matrix).max())
        if max_value > 1:
            matrix /= max_value

        eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
        negative = eigenvalues < 0

    return matrix


# Get the column information of the data
def get_columns_info(data):
    discrete_unique_values = {}
    col_names = []
    nan_cols = []
    dtype_int = []
    discrete_columns = []
    for column_name in data.columns:
        col_names.append(column_name)
        raw_column_data = data[column_name].values
        if data[column_name].dtype == "object":
            discrete_columns.append(column_name)
            discrete_unique_values[column_name] = np.unique(raw_column_data.astype("str"))
        else:
            no_nan = (1 - np.isnan(raw_column_data)).astype("bool")
            int_raw_column_data = raw_column_data[no_nan].astype("int32")
            if (int_raw_column_data == raw_column_data[no_nan]).sum() == len(
                raw_column_data[no_nan]
            ):
                dtype_int.append(column_name)
            if pd.isna(raw_column_data).sum() > 0:
                nan_cols.append(column_name)
    return discrete_unique_values, col_names, nan_cols, dtype_int, discrete_columns


# Aggregate the column information for each node
def merge_info(
    discrete_unique_values_list,
    col_names_list,
    nan_cols_list,
    dtype_int_list,
    discrete_columns_list,
):
    discrete_unique_values = {}
    nan_cols = np.array([])
    dtype_int = np.array(col_names_list[0])
    discrete_columns = np.array([])
    encoder_list = {}
    for i in range(len(nan_cols_list)):
        nan_cols = np.union1d(nan_cols, np.array(nan_cols_list[i]))
        discrete_columns = np.union1d(discrete_columns, np.array(discrete_columns_list[i])).tolist()
        dtype_int = np.intersect1d(dtype_int, np.array(dtype_int_list[i]))
    for item in col_names_list[0]:
        values = np.array([])
        for i in range(len(discrete_unique_values_list)):
            if item in discrete_columns:
                values = np.union1d(values, np.array(discrete_unique_values_list[i][item]))
        if len(values) > 0:
            discrete_unique_values[item] = values
        if item in discrete_columns:
            le = LabelEncoder()
            le.fit(discrete_unique_values[item])
            encoder_list[item] = le
    return (
        discrete_unique_values,
        list(nan_cols),
        list(dtype_int),
        list(discrete_columns),
        encoder_list,
    )


# Get the minimum value of the column data containing nan
def get_nan_columns_min(data, nan_cols):
    nan_columns_min = {}
    for column_name in data.columns:
        if column_name in nan_cols:
            raw_column_data = data[column_name].values
            nan_columns_min[column_name] = np.nanmin(raw_column_data)
    return nan_columns_min


# Aggregate the minimum value of the column data containing nan for each node
def merge_nan_columns_min(nan_columns_min_list, nan_cols):
    nan_columns_min = {}
    for item in nan_cols:
        min_values = []
        for i in range(len(nan_columns_min_list)):
            min_values.append(nan_columns_min_list[i][item])
        nan_columns_min[item] = np.array(min_values).min()
    return nan_columns_min


# Column data pre-processing
def process_columns_data(data, nan_columns_min, discrete_columns, encoder_list):
    dtypes = {}
    transform_data = []
    for column_name in data.columns:
        raw_column_data = data[column_name].values
        if column_name in discrete_columns:
            transform_column_data = (
                encoder_list[column_name].transform(raw_column_data.astype(str)).astype("float32")
            )
            dtypes[column_name] = str(transform_column_data.dtype)
        else:
            dtypes[column_name] = str(raw_column_data.dtype)
            if pd.isna(raw_column_data).sum() > 0:
                raw_column_data[pd.isna(raw_column_data)] = nan_columns_min[column_name] - 1
            transform_column_data = raw_column_data
        transform_data.append(transform_column_data)
    return np.array(transform_data, dtype=np.float64).T, dtypes


# Get the statistics of the column data
def get_columns_statistics(data, epsilon, h):
    column_min = data.min(axis=0)
    column_max = data.max(axis=0)
    count = len(data)
    if epsilon != 0:
        column_min = []
        column_max = []
        for i in range(data.shape[1]):
            f = lambda x: x + np.random.laplace(loc=0, scale=1 / (epsilon / 10 / h))
            column_count = pd.Series(data[:, i]).value_counts()
            min_counts = column_count.sort_index(ascending=False)
            max_counts = column_count.sort_index(ascending=True)
            min_counts = min_counts.cumsum() - min_counts.values
            min_counts = min_counts.apply(f)
            min_value = min_counts.index[min_counts.argmax()]
            max_counts = max_counts.cumsum() - max_counts.values
            max_counts = max_counts.apply(f)
            max_value = max_counts.index[max_counts.argmax()]
            if min_value >= max_value:
                max_value = min_value + 1e-6
            column_min.append(min_value)
            column_max.append(max_value)
        column_min = np.array(column_min)
        column_max = np.array(column_max)
        count = len(data) + np.random.laplace(loc=0, scale=1 / (epsilon / 10 / h))
    return column_min, column_max, count


# Aggregate the statistics of the column data for each node
def merge_statistics(column_min_list, column_max_list, count_list):
    column_min = np.array(column_min_list).min(axis=0)
    column_max = np.array(column_max_list).max(axis=0)
    count = sum(count_list)
    return column_min, column_max, count


# Compute the cumulative distribution table
def get_columns_cdf(data, column_min, column_max, t, epsilon, h):
    cdf_tables = []
    for i in range(data.shape[1]):
        cdf_table = {}
        n, bins = np.histogram(data[:, i], t, (column_min[i], column_max[i]), density=False)
        n = n.astype("float64")
        cdf_table["data"] = bins
        cdf_table["cdf"] = n
        if epsilon != 0:
            noise = np.random.laplace(loc=0, scale=1 / (epsilon / 2 / h), size=len(n))
            cdf_table["cdf"] += noise
        cdf_tables.append(cdf_table)
    return cdf_tables


# Aggregate the cumulative distribution table for each node
def merge_cdf(cdf_tables_list, count_list, t):
    cdf_tables = []
    for i in range(len(cdf_tables_list[0])):
        cdf_table = {}
        cdf_table["data"] = cdf_tables_list[0][i]["data"]
        cdf = np.zeros(len(cdf_tables_list[0][i]["cdf"]))
        for j in range(len(cdf_tables_list)):
            cdf += cdf_tables_list[j][i]["cdf"]
        zero_index = np.argwhere(cdf <= 0).reshape(-1)
        cdf[zero_index] = 1e-6
        cdf /= cdf.sum()
        cdf_table["cdf"] = np.cumsum(cdf)
        cdf_tables.append(cdf_table)
    return cdf_tables


# Transform the column data to the data following Gaussian distribution
def transform_data(data, cdf_tables, column_min, column_max, t):
    transformed_data = []
    for i in range(data.shape[1]):
        cdf_tables[i]["cdf"][-1] -= 1e-6
        flag = cdf_tables[i]["cdf"][0] == 0
        if flag:
            cdf_tables[i]["cdf"][0] += 1e-6
        transformed_data.append(
            norm.ppf(
                cdf_tables[i]["cdf"][
                    np.clip(
                        np.floor(
                            t
                            * (data[:, i] - column_min[i])
                            / (column_max[i] + 1e-6 - column_min[i])
                        ).astype("int32"),
                        0,
                        t - 1,
                    )
                ],
                0.0,
                1.0,
            )
        )
        if flag:
            cdf_tables[i]["cdf"][0] -= 1e-6
        cdf_tables[i]["cdf"][-1] += 1e-6
    return np.array(transformed_data).T


# Compute the inner product of column data
def get_inner_product_and_sum(data, epsilon, h):
    column_product = []
    if epsilon != 0:
        data = np.clip(data, -3, 3)
    for i in range(data.shape[1]):
        for j in range(i, data.shape[1]):
            dot = np.dot(data[:, i], data[:, j]).astype(np.float32)
            if epsilon != 0:
                dot += np.random.laplace(
                    loc=0, scale=9 / (2 * epsilon / 5 / h / (data.shape[1] * (data.shape[1] - 1)))
                )
            column_product.append(dot)
    column_sum = data.sum(axis=0)
    return column_product, column_sum


# Generate synthetic data
def synthesize_data(
    column_product_list,
    column_sum_list,
    cdf_tables,
    count,
    col_names,
    dtypes,
    nan_cols,
    nan_columns_min,
    dtype_int,
    discrete_columns,
    encoder_list,
    t,
    g,
):
    column_product = np.array(column_product_list).sum(axis=0)
    corration = np.zeros((len(cdf_tables), len(cdf_tables)), dtype="float32")
    k = 0

    # Compute covariance
    for i in range(len(cdf_tables)):
        for j in range(i, len(cdf_tables)):
            corration[i][j] = column_product[k] / (count - 1)
            if j != i:
                corration[j][i] = column_product[k] / (count - 1)
            k += 1
    corration = np.nan_to_num(corration, nan=0.0)

    # If singular, add some noise to the diagonal
    if np.linalg.cond(corration) > 1.0 / sys.float_info.epsilon:
        corration = corration + np.identity(corration.shape[0]) * 1e-6
    corration = get_nearest_correlation_matrix(corration)

    # Generate data that follows a multivariate Gaussian distribution
    Y = np.random.multivariate_normal([0] * corration.shape[0], corration, int(np.around(count)))

    tables = []
    new_data = []

    # Compute the inverse cumulative distribution table and transform the data that follow Gaussian distribution based on the inverse cumulative distribution table
    for i in range(corration.shape[1]):
        table = []
        bins = cdf_tables[i]["data"]
        n = cdf_tables[i]["cdf"]
        n[-1] = 1.0
        n = np.around(n, 5)
        old = 0
        table.append([bins[0], bins[1]])
        for j in range(len(n)):
            if n[j] == old:
                continue
            for k in range(int(old * g) + 1, int(n[j] * g) + 1):
                table.append([bins[j], bins[j + 1]])
            old = n[j]
        tables.append(table)
        cdf = norm.cdf(Y[:, i], Y[:, i].mean(), Y[:, i].std())
        values = []
        cdf = np.around(cdf, 5)
        for item in cdf:
            values.append(random.uniform(table[int(item * g)][0], table[int(item * g)][1]))
        values = np.array(values)
        new_data.append(values)

    new_data = np.array(new_data).T
    sys_data = pd.DataFrame(new_data, columns=col_names)

    # Synthetic column data post-processing
    for column_name in sys_data.columns:
        if column_name in discrete_columns:
            sys_data[column_name] = pd.Series(
                encoder_list[column_name].inverse_transform(
                    np.around(sys_data[column_name]).astype("int32")
                ),
                index=sys_data[column_name].index,
            )
            sys_data.loc[sys_data[column_name] == "nan", column_name] = np.nan
            sys_data[column_name] = sys_data[column_name].astype("object")
        else:
            if column_name in dtype_int:
                sys_data[column_name] = np.around(sys_data[column_name].astype("float64")).astype(
                    dtypes[column_name]
                )
            else:
                sys_data[column_name] = sys_data[column_name].astype(dtypes[column_name])
            if column_name in nan_cols:
                sys_data.loc[sys_data[column_name] < nan_columns_min[column_name], column_name] = (
                    np.nan
                )
    return sys_data, corration, tables


# Generate synthetic data in federated case
def federated_synthesize(data_list, t, g, epsilon):
    discrete_unique_values_list = []
    col_names_list = []
    nan_cols_list = []
    dtype_int_list = []
    discrete_columns_list = []
    h = len(data_list)
    for i in range(len(data_list)):
        discrete_unique_values, col_names, nan_cols, dtype_int, discrete_columns = get_columns_info(
            data_list[i]
        )
        discrete_unique_values_list.append(discrete_unique_values)
        col_names_list.append(col_names)
        nan_cols_list.append(nan_cols)
        dtype_int_list.append(dtype_int)
        discrete_columns_list.append(discrete_columns)

    discrete_unique_values, nan_cols, dtype_int, discrete_columns, encoder_list = merge_info(
        discrete_unique_values_list,
        col_names_list,
        nan_cols_list,
        dtype_int_list,
        discrete_columns_list,
    )

    nan_columns_min_list = []
    for i in range(len(data_list)):
        nan_columns_min = get_nan_columns_min(data_list[i], nan_cols)
        nan_columns_min_list.append(nan_columns_min)

    nan_columns_min = merge_nan_columns_min(nan_columns_min_list, nan_cols)

    processed_data_list = []
    dtypes_list = []
    for i in range(len(data_list)):
        processed_data, dtypes = process_columns_data(
            data_list[i], nan_columns_min, discrete_columns, encoder_list
        )
        processed_data_list.append(processed_data)
        dtypes_list.append(dtypes)

    column_min_list = []
    column_max_list = []
    count_list = []
    for i in range(len(processed_data_list)):
        column_min, column_max, count = get_columns_statistics(processed_data_list[i], epsilon, h)
        column_min_list.append(column_min)
        column_max_list.append(column_max)
        count_list.append(count)

    column_min, column_max, count = merge_statistics(column_min_list, column_max_list, count_list)

    cdf_tables_list = []
    for i in range(len(processed_data_list)):
        cdf_tables = get_columns_cdf(processed_data_list[i], column_min, column_max, t, epsilon, h)
        cdf_tables_list.append(cdf_tables)

    cdf_tables = merge_cdf(cdf_tables_list, count_list, t)

    column_product_list = []
    column_sum_list = []
    transform_data_list = []
    for i in range(len(processed_data_list)):
        transformed_data = transform_data(
            processed_data_list[i], cdf_tables, column_min, column_max, t
        )
        column_product, column_sum = get_inner_product_and_sum(transformed_data, epsilon, h)
        transform_data_list.append(transformed_data)
        column_product_list.append(column_product)
        column_sum_list.append(column_sum)

    sys_data, corration, tables = synthesize_data(
        column_product_list,
        column_sum_list,
        cdf_tables,
        count,
        col_names_list[0],
        dtypes_list[0],
        nan_cols,
        nan_columns_min,
        dtype_int,
        discrete_columns,
        encoder_list,
        t,
        g,
    )
    return sys_data
