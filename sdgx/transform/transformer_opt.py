import os
from collections import namedtuple
from io import StringIO
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

from sdgx.utils.io.csv_utils import get_csv_column, get_csv_column_count

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    ["column_name", "column_type", "transform", "output_info", "output_dimensions"],
)


# OPTIMIZE SDG 重写的 Data Transformer
class DataTransformer(object):
    """OPTIMIZE SDG 重写的 Data Transformer

    应对大数据（数据 > 内存）情况下的 Transformer 解决方案
        - 对于连续列：使用 BayesianGMM 对连续列建模并标准化为标量 [0, 1] 和向量。
        - 对于离散列：使用 scikit-learn OneHotEncoder 进行编码。

    本例的实现主要是为了支持大数据情况，并支持将 transform 的数据结果 以及 transformer
    所需的各类元数据 dump 到硬盘中，以实现较少的内存消耗。
    """

    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Create a data transformer.

        输入参数:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayesian GMM.
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
        """
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def _fit_continuous(self, column_name):
        """针对连续特征，训练一个贝叶斯 GMM， 用于 transform。
            该方法暂时不需要做进一步修改

        输入参数:
            data (pd.DataFrame):
                一个 pd.DataFrame 对象，但是只含有单列数据

        返回对象说明:
            namedtuple对象:
                返回单个 ``ColumnTransformInfo`` 对象
        """
        data = get_csv_column(self.raw_data_path, column_name)

        # column_name = data.columns[0]

        gm = ClusterBasedNormalizer(
            missing_value_generation="from_column", max_clusters=min(len(data), 10)
        )
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        # 使用 transform 对象存储单个列的 transformer
        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            output_info=[SpanInfo(1, "tanh"), SpanInfo(num_components, "softmax")],
            output_dimensions=1 + num_components,
        )

    def _fit_discrete(self, column_name):
        """针对离散特征：使用一个 one hot encoder 进行编码.
            该方法暂时不需要做进一步修改

        输入参数:
            data (pd.DataFrame):
                一个 pd.DataFrame 对象，但是只含有单列数据
            column_name (str)：
                需要进行 fit 操作的列名

        返回对象说明:
            namedtuple 对象:
                返回单个 ``ColumnTransformInfo`` 对象
        """
        data = get_csv_column(self.raw_data_path, column_name)
        # column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        # 使用 transform 对象存储单个列的 transformer
        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=ohe,
            output_info=[SpanInfo(num_categories, "softmax")],
            output_dimensions=num_categories,
        )

    # `fit` 是 transform 的入口方法
    def fit(self, raw_data_path, discrete_columns: Optional[List] = None, csv_cache_N=1000):
        """进行 DataTransformer 的 Fit 操作

        针对连续变量，使用 ``ClusterBasedNormalizer`` 进行 transform；
        对于离散变量，使用 ``OneHotEncoder``  进行 transform。

        此步骤还对矩阵数据和跨度信息中的#columns 进行计数操作。

        输入参数:
            raw_data_path (str):
                目前仅接受 csv 文件作为输入。
            discrete_columns (list):
                离散列名称，需要与 csv 文件的列名对应名，
                默认该参数为 [] 。
            csv_cache_N (int):
                为了减少内存，使用缓存 csv 的行数。
        """
        # raw data 需要从 path 中获取
        # raw_data = None
        self.raw_data_path = raw_data_path
        self.output_info_list = []
        self.output_dimensions = 0
        self.csv_cache_N = csv_cache_N
        # self.dataframe = False
        if not discrete_columns:
            discrete_columns = []

        # 对 data frame 的检测，如果不是 data frame ，还是转换为了 dataframe 进行操作
        # 但是经过搜索 此处仅对 后续一个逻辑又变化，所以我们直接把 self. dataframe 设置为 False
        # 在未来处理的时候，单独载入 data frame
        # if not isinstance(raw_data, pd.DataFrame):
        #     self. dataframe = False
        #     # work around for RDT issue #328 Fitting with numerical column names fails
        #     discrete_columns = [str(column) for column in discrete_columns]
        #     column_names = [str(num) for num in range(raw_data.shape[1])]
        #     raw_data = pd.DataFrame(raw_data, columns=column_names)

        # 对 discrete_columns 的处理还是保留了下来
        discrete_columns = [str(column) for column in discrete_columns]

        # 对 raw_data_path 增加检查
        #   - 是否存在，使用 os.path.exists 判断
        #   - 是否是 csv ，目前仅做名称上的检查
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError("Input csv NOT Found.")
        if not "csv" in raw_data_path:
            raise ValueError("Input Data is NOT CSV.")

        # raw data columns 在原有的实现方法中使用了 raw_data.columns
        # 这么做的问题就在于 df 直接载入了所有数据
        # 为了减少内存消耗，我们通过读取 csv 的前 x 行，并借助原有代码完成相应功能的计算
        cache_csv_str = ""
        with open(raw_data_path, "r") as f:
            for _ in range(self.csv_cache_N + 1):
                cache_csv_str += f.readline()
        # 从 str 创建 data frame
        cache_io = StringIO(cache_csv_str)
        raw_data_cache = pd.read_csv(cache_io)

        # get columns
        self.column_names = [str(num) for num in range(raw_data_cache.shape[1])]

        self._column_raw_dtypes = raw_data_cache.infer_objects().dtypes
        self._column_transform_info_list = []

        for column_name in raw_data_cache.columns:
            if column_name in discrete_columns:
                column_transform_info = self._fit_discrete(column_name)
            else:
                column_transform_info = self._fit_continuous(column_name)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f"{column_name}.normalized"].to_numpy()
        index = transformed[f"{column_name}.component"].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _synchronous_transform(self, input_data_path, column_transform_info_list, output_path):
        """Take a csv path, and transform columns synchronous.

        Outputs a list with Numpy arrays to disk.
        """

        loop = True
        # has_write_header = True
        # use iterator = True
        reader = pd.read_csv(input_data_path, iterator=True, chunksize=1000000)
        while loop:
            column_data_list = []
            # get raw data
            raw_data = None
            try:
                raw_data = reader.get_chunk()
            except StopIteration:
                loop = False

            # break if loop is iteration is end
            if not loop:
                break

            """
            # write csv header to file
            if has_write_header:
                f = open(output_path, "w")
                f.write(",".join(raw_data.columns.to_list()) + '\n')
                f.close()
                has_write_header = False
            """

            # transform data here
            for column_transform_info in column_transform_info_list:
                column_name = column_transform_info.column_name
                # 获取列数据
                data = raw_data[[column_name]]
                if column_transform_info.column_type == "continuous":
                    column_data_list.append(self._transform_continuous(column_transform_info, data))
                else:
                    column_data_list.append(self._transform_discrete(column_transform_info, data))

            # 追加写到 output path 吧
            chunk_array = np.concatenate(column_data_list, axis=1).astype(float)
            # file object
            f = open(output_path, "a")
            np.savetxt(f, chunk_array, fmt="%g", delimiter=",")
            f.close()
        # end while
        # return column_data_list

    def _parallel_transform(self, raw_data, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns in parallel.

        Outputs a list with Numpy arrays.
        """
        processes = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = raw_data[[column_name]]
            process = None
            if column_transform_info.column_type == "continuous":
                process = delayed(self._transform_continuous)(column_transform_info, data)
            else:
                process = delayed(self._transform_discrete)(column_transform_info, data)
            processes.append(process)

        return Parallel(n_jobs=-1)(processes)

    # TODO function to finish
    def transform(self, input_data_path, output_data_path):
        """Take raw data and output a matrix data."""
        # 从 disk 中读取
        raw_data = None

        #  check the parameter `input_data_path`
        if not os.path.exists(input_data_path):
            raise FileNotFoundError("Input csv NOT Found.")
        if not "csv" in input_data_path:
            raise ValueError("Input Data is NOT CSV.")

        # check the parameter `output data path`
        if not "csv" in output_data_path:
            raise ValueError("Input Data is NOT CSV.")

        # get the csv shape column count
        # import func from utils
        column_cnt = get_csv_column_count(input_data_path)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        # if column_cnt < 500:
        # column_data_list = self._synchronous_transform(
        self._synchronous_transform(
            input_data_path, self._column_transform_info_list, output_data_path
        )
        # else:
        #     column_data_list = self._parallel_transform(
        #         input_data_path,
        #         self._column_transform_info_list)

        # return np.concatenate(column_data_list, axis=1).astype(float)
        return output_data_path

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def inverse_transform(self, data, sigmas=None, output_path=None, write_header=True):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]
            if column_transform_info.column_type == "continuous":
                recovered_column_data = self._inverse_transform_continuous(
                    column_transform_info, column_data, sigmas, st
                )
            else:
                recovered_column_data = self._inverse_transform_discrete(
                    column_transform_info, column_data
                )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names).astype(
            self._column_raw_dtypes
        )

        # input is csv, self.dataframe will always be true
        # if not self.dataframe:
        #     recovered_data = recovered_data.to_numpy()
        if output_path is not None:
            f = open(output_path, "w")
            # write csv header to file
            if write_header:
                # f = open(output_path, "w")
                # recovered_data's type is pd.DataFrame
                f.write(",".join(recovered_data.columns.to_list()) + "\n")
            f.close()
            recovered_data.to_csv(output_path, index=False)

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Get the ids of the given `column_name`."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(f"The column_name `{column_name}` doesn't exist in the data.")

        ohe = column_transform_info.transform
        data = pd.DataFrame([value], columns=[column_transform_info.column_name])
        one_hot = ohe.transform(data).to_numpy()[0]
        if sum(one_hot) == 0:
            raise ValueError(f"The value `{value}` doesn't exist in the column `{column_name}`.")

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot),
        }
