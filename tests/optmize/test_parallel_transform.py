import os
import warnings

import numpy as np

from sdgx.data_connectors.csv_connector import CsvConnector
from sdgx.data_loader import DataLoader
from sdgx.data_models.metadata import Metadata
from sdgx.models.components.optimize.sdv_ctgan.data_sampler import DataSampler
from sdgx.models.components.optimize.sdv_ctgan.data_transformer import DataTransformer


def custom_warning_handler(e, category, filename, lineno, file=None, line=None):
    print(f"Caught warning: {e}")
    assert repr(e).find("invalid value encountered in log") == -1


warnings.showwarning = custom_warning_handler


def preparing_data():
    # Same code with CTGAN prefit.
    dataset_csv = os.path.join(os.path.dirname(__file__), "data_for_test_parallel_transform.csv")
    data_connector = CsvConnector(path=dataset_csv)
    data_loader = DataLoader(data_connector)
    data_metadata = Metadata.from_dataloader(data_loader)
    data_metadata.datetime_format = {key: "%Y/%m/%d" for key in data_metadata.datetime_columns}
    transformer = DataTransformer()
    transformer.fit(data_loader, data_metadata.discrete_columns)
    return transformer, data_loader


# def unfixed_parallel_transform(self, raw_data, column_transform_info_list):
#     processes = []
#     for column_transform_info in column_transform_info_list:
#         column_name = column_transform_info.column_name
#         data = raw_data[[column_name]]
#         process = None
#         if column_transform_info.column_type == "continuous":
#             process = delayed(self._transform_continuous)(column_transform_info, data)
#         else:
#             process = delayed(self._transform_discrete)(column_transform_info, data)
#         processes.append(process)
#
#     p = Parallel(n_jobs=-1, return_as="generator_unordered")
#     loader = NDArrayLoader()
#     for ndarray in p(processes):
#         loader.store(ndarray.astype(float))
#     return loader


def test_parallel_transform_fixed():
    transformer, data_loader = preparing_data()
    # start test
    ndarry_loader = transformer._parallel_transform(
        data_loader, transformer._column_transform_info_list
    )
    data_sampler = DataSampler(ndarry_loader, transformer.output_info_list, True)


# def test_parallel_transform_unfixed():
#     transformer, data_loader = preparing_data()
#     # start test
#     ndarry_loader = unfixed_parallel_transform(transformer, data_loader, transformer._column_transform_info_list)
#     data_sampler = DataSampler(ndarry_loader, transformer.output_info_list, True)


def find_not_matching_column_type(data, column_transform_info_list):
    col_index = 0
    for column_transform_info in column_transform_info_list:
        output_dim = column_transform_info.output_dimensions
        if column_transform_info.column_type == "discrete":
            arr = data[:, col_index : col_index + output_dim]
            # if bug occurred, the arr is switched as continuous
            print(
                f"Filter not one-hot data for column {column_transform_info.column_name}: ",
                arr[(arr != 0) & (arr != 1)],
            )
            assert np.all((arr == 0) | (arr == 1))
        col_index += output_dim


def test_parallel_transform_fixed_not_columns_switching():
    transformer, data_loader = preparing_data()
    ndarry_loader = transformer._parallel_transform(
        data_loader, transformer._column_transform_info_list
    )

    find_not_matching_column_type(ndarry_loader, transformer._column_transform_info_list)


# def test_parallel_transform_unfixed_caused_columns_switching():
#     unfixed_transformer, unfixed_data_loader = preparing_data()
#     unfixed_ndarry_loader = unfixed_parallel_transform(unfixed_transformer, unfixed_data_loader,
#                                                        unfixed_transformer._column_transform_info_list)
#
#     find_not_matching_column_type(unfixed_ndarry_loader, unfixed_transformer._column_transform_info_list)
