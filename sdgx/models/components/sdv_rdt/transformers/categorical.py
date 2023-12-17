"""Transformers for categorical data."""

import warnings

import numpy as np
import pandas as pd
import psutil
from scipy.stats import norm

from sdgx.models.components.sdv_rdt.errors import Error
from sdgx.models.components.sdv_rdt.transformers.base import BaseTransformer


class FrequencyEncoder(BaseTransformer):
    """Transformer for categorical data.

    This transformer computes a float representative for each one of the categories
    found in the fit data, and then replaces the instances of these categories with
    the corresponding representative.

    The representatives are decided by sorting the categorical values by their relative
    frequency, then dividing the ``[0, 1]`` interval by these relative frequencies, and
    finally assigning the middle point of each interval to the corresponding category.

    When the transformation is reverted, each value is assigned the category that
    corresponds to the interval it falls in.

    Null values are considered just another category.

    Args:
        add_noise (bool):
            Whether to generate gaussian noise around the class representative of each interval
            or just use the mean for all the replaced values. Defaults to ``False``.
    """

    INPUT_SDTYPE = "categorical"
    SUPPORTED_SDTYPES = ["categorical", "boolean"]
    OUTPUT_SDTYPES = {"value": "float"}
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    mapping = None
    intervals = None
    starts = None
    means = None
    dtype = None

    def __setstate__(self, state):
        """Replace any ``null`` key by the actual ``np.nan`` instance."""
        intervals = state.get("intervals")
        if intervals:
            for key in list(intervals):
                if pd.isna(key):
                    intervals[np.nan] = intervals.pop(key)

        self.__dict__ = state

    def __init__(self, add_noise=False):
        self.add_noise = add_noise

    def is_transform_deterministic(self):
        """Return whether the transform is deterministic.

        Returns:
            bool:
                Whether or not the transform is deterministic.
        """
        return not self.add_noise

    @staticmethod
    def _get_intervals(data):
        """Compute intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to analyze.

        Returns:
            dict:
                intervals for each categorical value (start, end).
        """
        data = data.fillna(np.nan)
        frequencies = data.value_counts(dropna=False)

        start = 0
        end = 0
        elements = len(data)

        intervals = {}
        means = []
        starts = []
        for value, frequency in frequencies.items():
            prob = frequency / elements
            end = start + prob
            mean = start + prob / 2
            std = prob / 6
            if pd.isna(value):
                value = np.nan

            intervals[value] = (start, end, mean, std)
            means.append(mean)
            starts.append((value, start))
            start = end

        means = pd.Series(means, index=list(frequencies.keys()))
        starts = pd.DataFrame(starts, columns=["category", "start"]).set_index("start")

        return intervals, means, starts

    def _fit(self, data):
        """Fit the transformer to the data.

        Compute the intervals for each categorical value.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self.dtype = data.dtype
        self.intervals, self.means, self.starts = self._get_intervals(data)

    @staticmethod
    def _clip_noised_transform(result, start, end):
        """Clip transformed values.

        Used to ensure the noise added to transformed values doesn't make it
        go out of the bounds of a given category.

        The upper bound must be slightly lower than ``end``
        so it doesn't get treated as the next category.
        """
        return np.clip(result, start, end - 1e-9)

    def _transform_by_category(self, data):
        """Transform the data by iterating over the different categories."""
        result = np.empty(shape=(len(data),), dtype=float)

        # loop over categories
        for category, values in self.intervals.items():
            start, end, mean, std = values
            if category is np.nan:
                mask = data.isna()
            else:
                mask = data.to_numpy() == category

            if self.add_noise:
                result[mask] = norm.rvs(mean, std, size=mask.sum())
                result[mask] = self._clip_noised_transform(result[mask], start, end)
            else:
                result[mask] = mean

        return result

    def _get_value(self, category):
        """Get the value that represents this category."""
        if pd.isna(category):
            category = np.nan

        start, end, mean, std = self.intervals[category]

        if self.add_noise:
            result = norm.rvs(mean, std)
            return self._clip_noised_transform(result, start, end)

        return mean

    def _transform_by_row(self, data):
        """Transform the data row by row."""
        return data.fillna(np.nan).apply(self._get_value).to_numpy()

    def _transform(self, data):
        """Transform the categorical values to float representatives.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        fit_categories = pd.Series(self.intervals.keys())
        has_nan = pd.isna(fit_categories).any()
        unseen_indexes = ~(data.isin(fit_categories) | (pd.isna(data) & has_nan))
        if unseen_indexes.any():
            # Select only the first 5 unseen categories to avoid flooding the console.
            unseen_categories = set(data[unseen_indexes][:5])
            warnings.warn(
                f"The data contains {unseen_indexes.sum()} new categories that were not "
                f"seen in the original data (examples: {unseen_categories}). Assigning "
                "them random values. If you want to model new categories, "
                "please fit the transformer again with the new data."
            )

        data[unseen_indexes] = np.random.choice(fit_categories, size=unseen_indexes.size)
        if len(self.means) < len(data):
            return self._transform_by_category(data)

        return self._transform_by_row(data)

    def _reverse_transform_by_matrix(self, data):
        """Reverse transform the data with matrix operations."""
        num_rows = len(data)
        num_categories = len(self.starts)

        data = np.broadcast_to(data, (num_categories, num_rows)).T
        starts = np.broadcast_to(self.starts.index, (num_rows, num_categories))
        is_data_greater_than_starts = (data >= starts)[:, ::-1]
        interval_indexes = num_categories - np.argmax(is_data_greater_than_starts, axis=1) - 1

        get_category_from_index = list(self.starts["category"]).__getitem__
        return pd.Series(interval_indexes).apply(get_category_from_index).astype(self.dtype)

    def _reverse_transform_by_category(self, data):
        """Reverse transform the data by iterating over all the categories."""
        result = np.empty(shape=(len(data),), dtype=self.dtype)

        # loop over categories
        for category, values in self.intervals.items():
            start = values[0]
            mask = start <= data.to_numpy()
            result[mask] = category

        return pd.Series(result, index=data.index, dtype=self.dtype)

    def _get_category_from_start(self, value):
        lower = self.starts.loc[:value]
        return lower.iloc[-1].category

    def _reverse_transform_by_row(self, data):
        """Reverse transform the data by iterating over each row."""
        return data.apply(self._get_category_from_start).astype(self.dtype)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series):
                Data to revert.

        Returns:
            pandas.Series
        """
        data = data.clip(0, 1)
        num_rows = len(data)
        num_categories = len(self.means)

        # total shape * float size * number of matrices needed
        needed_memory = num_rows * num_categories * 8 * 3
        available_memory = psutil.virtual_memory().available
        if available_memory > needed_memory:
            return self._reverse_transform_by_matrix(data)

        if num_rows > num_categories:
            return self._reverse_transform_by_category(data)

        # loop over rows
        return self._reverse_transform_by_row(data)


class OneHotEncoder(BaseTransformer):
    """OneHotEncoding for categorical data.

    This transformer replaces a single vector with N unique categories in it
    with N vectors which have 1s on the rows where the corresponding category
    is found and 0s on the rest.

    Null values are considered just another category.
    """

    INPUT_SDTYPE = "categorical"
    SUPPORTED_SDTYPES = ["categorical", "boolean"]
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    dummies = None
    _dummy_na = None
    _num_dummies = None
    _dummy_encoded = False
    _indexer = None
    _uniques = None

    @staticmethod
    def _prepare_data(data):
        """Transform data to appropriate format.

        If data is a valid list or a list of lists, transforms it into an np.array,
        otherwise returns it.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to prepare.

        Returns:
            pandas.Series or numpy.ndarray
        """
        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError("Unexpected format.")
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError("Unexpected format.")

            data = data[:, 0]

        return data

    def get_output_sdtypes(self):
        """Return the output sdtypes produced by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        output_sdtypes = {f"value{i}": "float" for i in range(len(self.dummies))}

        return self._add_prefix(output_sdtypes)

    def _fit(self, data):
        """Fit the transformer to the data.

        Get the pandas `dummies` which will be used later on for OneHotEncoding.

        Args:
            data (pandas.Series or pandas.DataFrame):
                Data to fit the transformer to.
        """
        data = self._prepare_data(data)

        null = pd.isna(data)
        self._uniques = list(pd.unique(data[~null]))
        self._dummy_na = null.any()
        self._num_dummies = len(self._uniques)
        self._indexer = list(range(self._num_dummies))
        self.dummies = self._uniques.copy()

        if not np.issubdtype(data.dtype, np.number):
            self._dummy_encoded = True

        if self._dummy_na:
            self.dummies.append(np.nan)

    def _transform_helper(self, data):
        if self._dummy_encoded:
            coder = self._indexer
            codes = pd.Categorical(data, categories=self._uniques).codes
        else:
            coder = self._uniques
            codes = data

        rows = len(data)
        dummies = np.broadcast_to(coder, (rows, self._num_dummies))
        coded = np.broadcast_to(codes, (self._num_dummies, rows)).T
        array = (coded == dummies).astype(int)

        if self._dummy_na:
            null = np.zeros((rows, 1), dtype=int)
            null[pd.isna(data)] = 1
            array = np.append(array, null, axis=1)

        return array

    def _transform(self, data):
        """Replace each category with the OneHot vectors.

        Args:
            data (pandas.Series, list or list of lists):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._prepare_data(data)
        unique_data = {np.nan if pd.isna(x) else x for x in pd.unique(data)}
        unseen_categories = unique_data - set(self.dummies)
        if unseen_categories:
            # Select only the first 5 unseen categories to avoid flooding the console.
            examples_unseen_categories = set(list(unseen_categories)[:5])
            warnings.warn(
                f"The data contains {len(unseen_categories)} new categories that were not "
                f"seen in the original data (examples: {examples_unseen_categories}). Creating "
                "a vector of all 0s. If you want to model new categories, "
                "please fit the transformer again with the new data."
            )

        return self._transform_helper(data)

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)

        return pd.Series(indices).map(self.dummies.__getitem__)


class LabelEncoder(BaseTransformer):
    """LabelEncoding for categorical data.

    This transformer generates a unique integer representation for each category
    and simply replaces each category with its integer value.

    Null values are considered just another category.

    Attributes:
        values_to_categories (dict):
            Dictionary that maps each integer value for its category.
        categories_to_values (dict):
            Dictionary that maps each category with the corresponding
            integer value.

    Args:
        add_noise (bool):
            Whether to generate uniform noise around the label for each category.
            Defaults to ``False``.
        order_by (None or str):
            A string defining how to order the categories before assigning them labels. Defaults to
            ``None``. Options include:
            - ``'numerical_value'``: Order the categories by numerical value.
            - ``'alphabetical'``: Order the categories alphabetically.
            - ``None``: Use the order that the categories appear in when fitting.
    """

    INPUT_SDTYPE = "categorical"
    SUPPORTED_SDTYPES = ["categorical", "boolean"]
    OUTPUT_SDTYPES = {"value": "float"}
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    values_to_categories = None
    categories_to_values = None

    def __init__(self, add_noise=False, order_by=None):
        self.add_noise = add_noise
        if order_by not in [None, "alphabetical", "numerical_value"]:
            raise Error(
                "order_by must be one of the following values: None, 'numerical_value' or "
                "'alphabetical'"
            )

        self.order_by = order_by

    def _order_categories(self, unique_data):
        if self.order_by == "alphabetical":
            if unique_data.dtype.type not in [np.str_, np.object_]:
                raise Error("The data must be of type string if order_by is 'alphabetical'.")

        elif self.order_by == "numerical_value":
            if not np.issubdtype(unique_data.dtype.type, np.number):
                raise Error("The data must be numerical if order_by is 'numerical_value'.")

        if self.order_by is not None:
            nans = pd.isna(unique_data)
            unique_data = np.sort(unique_data[~nans])
            if nans.any():
                unique_data = np.append(unique_data, [np.nan])

        return unique_data

    def _fit(self, data):
        """Fit the transformer to the data.

        Generate a unique integer representation for each category and
        store them in the ``categories_to_values`` dict and its reverse
        ``values_to_categories``.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        unique_data = pd.unique(data.fillna(np.nan))
        unique_data = self._order_categories(unique_data)
        self.values_to_categories = dict(enumerate(unique_data))
        self.categories_to_values = {
            category: value for value, category in self.values_to_categories.items()
        }

    def _transform(self, data):
        """Replace each category with its corresponding integer value.

        If a category has not been seen before, a random value is assigned.

        If ``add_noise`` is True, the integer values will be replaced by a
        random number between the value and the value + 1.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            pd.Series
        """
        mapped = data.fillna(np.nan).map(self.categories_to_values)
        is_null = mapped.isna()
        if is_null.any():
            # Select only the first 5 unseen categories to avoid flooding the console.
            unseen_categories = set(data[is_null][:5])
            warnings.warn(
                f"The data contains {is_null.sum()} new categories that were not "
                f"seen in the original data (examples: {unseen_categories}). Assigning "
                "them random values. If you want to model new categories, "
                "please fit the transformer again with the new data."
            )

        mapped[is_null] = np.random.randint(len(self.categories_to_values), size=is_null.sum())

        if self.add_noise:
            mapped = np.random.uniform(mapped, mapped + 1)

        return mapped

    def _reverse_transform(self, data):
        """Convert float values back to the original categorical values.

        Args:
            data (pd.Series or numpy.ndarray):
                Data to revert.

        Returns:
            pandas.Series
        """
        if self.add_noise:
            data = np.floor(data)

        data = data.clip(min(self.values_to_categories), max(self.values_to_categories))
        return data.round().map(self.values_to_categories)


class CustomLabelEncoder(LabelEncoder):
    """Custom label encoder for categorical data.

    This class works very similarly to the ``LabelEncoder``, except that it requires the ordering
    for the labels to be provided.

    Null values are considered just another category.

    Args:
        order (list):
            A list of all the unique categories for the data. The order of the list determines the
            label that each category will get.
        add_noise (bool):
            Whether to generate uniform noise around the label for each category.
            Defaults to ``False``.
    """

    def __init__(self, order, add_noise=False):
        self.order = pd.Series(order).fillna(np.nan)
        super().__init__(add_noise=add_noise)

    def _fit(self, data):
        """Fit the transformer to the data.

        Generate a unique integer representation for each category and
        store them in the ``categories_to_values`` dict and its reverse
        ``values_to_categories``.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        data = data.fillna(np.nan)
        missing = list(data[~data.isin(self.order)].unique())
        if len(missing) > 0:
            raise Error(
                f"Unknown categories '{missing}'. All possible categories must be defined in the "
                "'order' parameter."
            )

        self.values_to_categories = dict(enumerate(self.order))
        self.categories_to_values = {
            category: value for value, category in self.values_to_categories.items()
        }
