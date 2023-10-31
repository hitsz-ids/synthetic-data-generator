"""
    Wrappers around copulas models.
    需要修改: fit接口以适应性能优化措施
"""
import logging
import warnings
from copy import deepcopy

import copulas
import copulas.univariate
import numpy as np
import pandas as pd
import scipy
from copulas import multivariate
from rdt.transformers import OneHotEncoder

from sdgx.errors import NonParametricError
from sdgx.statistics.base import BaseSynthesizerModel

# transformer 以及 sampler 已经拆分，挪到 transform/ 目录中
# from sdgx.transform.sampler import DataSamplerCTGAN
from sdgx.transform.transformer import DataTransformerCTGAN
from sdgx.utils.utils import (
    flatten_dict,
    log_numerical_distributions_error,
    unflatten_dict,
    validate_numerical_distributions,
)

LOGGER = logging.getLogger(__name__)


class GaussianCopulaSynthesizer(BaseSynthesizerModel):
    """Model wrapping ``copulas.multivariate.GaussianMultivariate`` copula.

    Args:
        metadata (sdv.metadata.SingleTableMetadata):
            Single table metadata representing the data that this synthesizer will be used for.
        enforce_min_max_values (bool):
            Specify whether or not to clip the data returned by ``reverse_transform`` of
            the numerical transformer, ``FloatFormatter``, to the min and max values seen
            during ``fit``. Defaults to ``True``.
        enforce_rounding (bool):
            Define rounding scheme for ``numerical`` columns. If ``True``, the data returned
            by ``reverse_transform`` will be rounded as in the original data. Defaults to ``True``.
        locales (list or str):
            The default locale(s) to use for AnonymizedFaker transformers. Defaults to ``None``.
        numerical_distributions (dict):
            Dictionary that maps field names from the table that is being modeled with
            the distribution that needs to be used. The distributions can be passed as either
            a ``copulas.univariate`` instance or as one of the following values:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a truncnorm distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.

        default_distribution (str):
            Copulas univariate distribution to use by default. Valid options are:

                * ``norm``: Use a norm distribution.
                * ``beta``: Use a Beta distribution.
                * ``truncnorm``: Use a Truncated Gaussian distribution.
                * ``uniform``: Use a uniform distribution.
                * ``gamma``: Use a Gamma distribution.
                * ``gaussian_kde``: Use a GaussianKDE distribution. This model is non-parametric,
                  so using this will make ``get_parameters`` unusable.
             Defaults to ``beta``.
    """

    # 这里代表的是几种可选的分布，这里通常使用 beta
    # 其他分布的内容我们可以以后再了解。
    # 这个应该是属于 类变量的
    _DISTRIBUTIONS = {
        "norm": copulas.univariate.GaussianUnivariate,
        "beta": copulas.univariate.BetaUnivariate,
        "truncnorm": copulas.univariate.TruncatedGaussian,
        "gamma": copulas.univariate.GammaUnivariate,
        "uniform": copulas.univariate.UniformUnivariate,
        "gaussian_kde": copulas.univariate.GaussianKDE,
    }

    # model 变量就是承载模型的，使用 self._model 来访问
    _model = None

    # 这个方法很简单，就是把上面 _distribution 给 return 出来
    # 相当于一个字典的作用
    @classmethod
    def get_distribution_class(cls, distribution):
        """Return the corresponding distribution class from ``copulas.univariate``.

        Args:
            distribution (str):
                A string representing a copulas univariate distribution.

        Returns:
            copulas.univariate:
                A copulas univariate class that corresponds to the distribution.
        """
        if not isinstance(distribution, str) or distribution not in cls._DISTRIBUTIONS:
            error_message = f"Invalid distribution specification '{distribution}'."
            raise ValueError(error_message)

        return cls._DISTRIBUTIONS[distribution]

    # 初始化方法，参数的类型需要仔细分析一下
    def __init__(
        self,
        metadata,
        enforce_min_max_values=True,
        enforce_rounding=True,
        locales=None,
        numerical_distributions=None,
        default_distribution=None,
    ):
        self.metadata = (metadata,)  # fit 数据集的元数据
        self.enforce_min_max_values = (enforce_min_max_values,)  # 限制最大最小value，一般都是true
        self.enforce_rounding = (enforce_rounding,)  # 保持相同的小数点位数， 一般都是true
        self.locales = (locales,)  # 和语言设置有关，这个暂时不太清楚

        # 验证分布？
        # 这个时候还没有输入数据的，只有 metadata
        # 应该是通过 metadata 来验证分布的
        validate_numerical_distributions(numerical_distributions, self.metadata)
        # 这里的意思是，如果没有指定分布，那么就使用 beta 分布
        # 下面分别是两个参数，重新赋值
        self.numerical_distributions = numerical_distributions or {}
        self.default_distribution = default_distribution or "beta"

        # 以下几行代码是使用 class method 把字符串类型的分布
        # 转化为 copulas.univariate 的分布实例
        self._default_distribution = self.get_distribution_class(self.default_distribution)
        self._numerical_distributions = {
            field: self.get_distribution_class(distribution)
            for field, distribution in self.numerical_distributions.items()
        }
        # 最后是行数，目前是 none
        # 目前是没有训练数据的
        # 这个变量可能和训练数据有关
        self._num_rows = None

    # 训练函数，输入的是训练数据
    # 需要是 data frame 格式
    # 初步的性能优化方案可以是：
    # 1. 增加增量学习机制，防止内存消耗过多
    # 2. 增加
    def fit(self, processed_data):
        # 载入 transformer
        self._transformer = DataTransformerCTGAN()
        self._transformer.fit(processed_data, self.metadata[0])

        # 使用 transformer 处理数据
        processed_data = pd.DataFrame(self._transformer.transform(processed_data))

        # 这个应该是打 log ，不影响实际训练
        log_numerical_distributions_error(
            self.numerical_distributions, processed_data.columns, LOGGER
        )

        # 记录学习数据的长度
        self._num_rows = len(processed_data)

        # 这里是把没有指定分布的列，都设置为默认分布
        numerical_distributions = deepcopy(self._numerical_distributions)
        for column in processed_data.columns:
            if column not in numerical_distributions:
                numerical_distributions[column] = self._numerical_distributions.get(
                    column, self._default_distribution
                )

        # 然后建立模型
        # 模型直接调用的 multivariate.GaussianMultivariate
        # 这个GaussianMultivariate是可以直接 fit 的，只不过不在这里
        self._model = multivariate.GaussianMultivariate(distribution=numerical_distributions)

        # 在这里调用 fit 方法
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="scipy")
            self._model.fit(processed_data)
        # 注意看，这里只是忽略了警告，但是是直接调用的 data frame 用于 fit
        # 所以从内存角度而言，多大的数据都是直接载入到内存中的，优化需要从这里下手

    def _warn_for_update_transformers(self, column_name_to_transformer):
        """Raise warnings for update_transformers.

        Args:
            column_name_to_transformer (dict):
                Dict mapping column names to transformers to be used for that column.
        """
        for column, transformer in column_name_to_transformer.items():
            sdtype = self.metadata.columns[column]["sdtype"]
            if sdtype == "categorical" and isinstance(transformer, OneHotEncoder):
                warnings.warn(
                    f"Using a OneHotEncoder transformer for column '{column}' "
                    "may slow down the preprocessing and modeling times."
                )

    # 生成数据函数，这里其实直接调用的 model 的方法
    # 没有其他了

    def sample(self, num_rows, conditions=None):
        """Sample the indicated number of rows from the model.

        Args:
            num_rows (int):
                Amount of rows to sample.
            conditions (dict):
                If specified, this dictionary maps column names to the column
                value. Then, this method generates ``num_rows`` samples, all of
                which are conditioned on the given variables.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        return self._transformer.inverse_transform(
            self._model.sample(num_rows, conditions=conditions).to_numpy()
        )

    def _get_valid_columns_from_metadata(self, columns):
        valid_columns = []
        for column in columns:
            for valid_column in self.metadata.columns:
                if column.startswith(valid_column):
                    valid_columns.append(column)
                    break

        return valid_columns

    def get_learned_distributions(self):
        """Get the marginal distributions used by the ``GaussianCopula``.

        Return a dictionary mapping the column names with the distribution name and the learned
        parameters for those.

        Returns:
            dict:
                Dictionary containing the distributions used or detected for each column and the
                learned parameters for those.
        """
        if not self._fitted:
            raise ValueError(
                "Distributions have not been learned yet. Please fit your model first using 'fit'."
            )

        parameters = self._model.to_dict()
        columns = parameters["columns"]
        univariates = deepcopy(parameters["univariates"])
        learned_distributions = {}
        valid_columns = self._get_valid_columns_from_metadata(columns)
        for column, learned_params in zip(columns, univariates):
            if column in valid_columns:
                distribution = self.numerical_distributions.get(column, self.default_distribution)
                learned_params.pop("type")
                learned_distributions[column] = {
                    "distribution": distribution,
                    "learned_parameters": learned_params,
                }

        return learned_distributions

    def _get_parameters(self):
        """Get copula model parameters.

        Compute model ``correlation`` and ``distribution.std``
        before it returns the flatten dict.

        Returns:
            dict:
                Copula parameters.

        Raises:
            NonParametricError:
                If a non-parametric distribution has been used.
        """
        for univariate in self._model.univariates:
            univariate_type = type(univariate)
            if univariate_type is copulas.univariate.Univariate:
                univariate = univariate._instance

            if univariate.PARAMETRIC == copulas.univariate.ParametricType.NON_PARAMETRIC:
                raise NonParametricError("This GaussianCopula uses non parametric distributions")

        params = self._model.to_dict()

        correlation = []
        for index, row in enumerate(params["correlation"][1:]):
            correlation.append(row[: index + 1])

        params["correlation"] = correlation
        params["univariates"] = dict(zip(params.pop("columns"), params["univariates"]))
        params["num_rows"] = self._num_rows

        return flatten_dict(params)

    @staticmethod
    def _get_nearest_correlation_matrix(matrix):
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

        iterations = 0
        while np.any(negative):
            eigenvalues[negative] = 0
            matrix = eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T)
            if iterations >= 10:
                break

            matrix = matrix - matrix * identity + identity

            max_value = np.abs(np.abs(matrix).max())
            if max_value > 1:
                matrix /= max_value

            eigenvalues, eigenvectors = scipy.linalg.eigh(matrix)
            negative = eigenvalues < 0
            iterations += 1

        return matrix

    @classmethod
    def _rebuild_correlation_matrix(cls, triangular_correlation):
        """Rebuild a valid correlation matrix from its lower half triangle.

        The input of this function is a list of lists of floats of size 1, 2, 3...n-1:

           [[c_{2,1}], [c_{3,1}, c_{3,2}], ..., [c_{n,1},...,c_{n,n-1}]]

        Corresponding to the values from the lower half of the original correlation matrix,
        **excluding** the diagonal.

        The output is the complete correlation matrix reconstructed using the given values
        and scaled to the :math:`[-1, 1]` range if necessary.

        Args:
            triangle_correlation (list[list[float]]):
                A list that contains lists of floats of size 1, 2, 3... up to ``n-1``,
                where ``n`` is the size of the target correlation matrix.

        Returns:
            numpy.ndarray:
                rebuilt correlation matrix.
        """
        zero = [0.0]
        size = len(triangular_correlation) + 1
        left = np.zeros((size, size))
        right = np.zeros((size, size))
        for idx, values in enumerate(triangular_correlation):
            values = values + zero * (size - idx - 1)
            left[idx + 1, :] = values
            right[:, idx + 1] = values

        correlation = left + right
        max_value = np.abs(correlation).max()
        if max_value > 1:
            correlation /= max_value

        correlation += np.identity(size)

        return cls._get_nearest_correlation_matrix(correlation).tolist()

    def _rebuild_gaussian_copula(self, model_parameters):
        """Rebuild the model params to recreate a Gaussian Multivariate instance.

        Args:
            model_parameters (dict):
                Sampled and reestructured model parameters.

        Returns:
            dict:
                Model parameters ready to recreate the model.
        """
        columns = []
        univariates = []
        for column, univariate in model_parameters["univariates"].items():
            columns.append(column)
            univariate["type"] = self.get_distribution_class(
                self._numerical_distributions.get(column, self.default_distribution)
            )
            if "scale" in univariate:
                univariate["scale"] = max(0, univariate["scale"])

            univariates.append(univariate)

        model_parameters["univariates"] = univariates
        model_parameters["columns"] = columns

        correlation = model_parameters.get("correlation")
        if correlation:
            model_parameters["correlation"] = self._rebuild_correlation_matrix(correlation)
        else:
            model_parameters["correlation"] = [[1.0]]

        return model_parameters

    def _get_likelihood(self, table_rows):
        return self._model.probability_density(table_rows)

    def _set_parameters(self, parameters):
        """Set copula model parameters.

        Args:
            dict:
                Copula flatten parameters.
        """
        parameters = unflatten_dict(parameters)
        if "num_rows" in parameters:
            num_rows = parameters.pop("num_rows")
            self._num_rows = 0 if pd.isna(num_rows) else max(0, int(round(num_rows)))

        if parameters:
            parameters = self._rebuild_gaussian_copula(parameters)
            self._model = multivariate.GaussianMultivariate.from_dict(parameters)
