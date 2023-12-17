"""Transformer for data that contains Null values."""

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class NullTransformer:
    """Transformer for data that contains Null values.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an integer, float or string is given,
            replace them with the given value. If the strings ``'mean'`` or ``'mode'`` are given,
            replace them with the corresponding aggregation (``'mean'`` only works for numerical
            values). If ``None`` is given, do not replace them. Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """

    nulls = None
    _model_missing_values = None
    _missing_value_replacement = None
    _null_percentage = None

    def __init__(self, missing_value_replacement=None, model_missing_values=False):
        self._missing_value_replacement = missing_value_replacement
        self._model_missing_values = model_missing_values

    def models_missing_values(self):
        """Indicate whether this transformer creates a null column on transform.

        Returns:
            bool:
                Whether a null column is created on transform.
        """
        return self._model_missing_values

    def _get_missing_value_replacement(self, data):
        """Get the fill value to use for the given data.

        Args:
            data (pd.Series):
                The data that is being transformed.

        Return:
            object:
                The fill value that needs to be used.
        """
        if self._missing_value_replacement is None:
            return None

        if self._missing_value_replacement == "mean":
            return data.mean()

        if self._missing_value_replacement == "mode":
            return data.mode(dropna=True)[0]

        return self._missing_value_replacement

    def fit(self, data):
        """Fit the transformer to the data.

        Evaluate if the transformer has to create the null column or not.

        Args:
            data (pandas.Series):
                Data to transform.
        """
        null_values = data.isna().to_numpy()
        self.nulls = null_values.any()

        self._missing_value_replacement = self._get_missing_value_replacement(data)
        if not self.nulls and self._model_missing_values:
            self._model_missing_values = False
            guidance_message = (
                f"Guidance: There are no missing values in column {data.name}. "
                "Extra column not created."
            )
            LOGGER.info(guidance_message)

        if not self._model_missing_values:
            self._null_percentage = null_values.sum() / len(data)

    def transform(self, data):
        """Replace null values with the indicated ``missing_value_replacement``.

        If required, create the null indicator column.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        isna = data.isna()
        if isna.any() and self._missing_value_replacement is not None:
            data = data.fillna(self._missing_value_replacement)

        if self._model_missing_values:
            return pd.concat([data, isna.astype(np.float64)], axis=1).to_numpy()

        return data.to_numpy()

    def reverse_transform(self, data):
        """Restore null values to the data.

        If a null indicator column was created during fit, use it as a reference.
        Otherwise, randomly replace values with ``np.nan``. The percentage of values
        that will be replaced is the percentage of null values seen in the fitted data.

        Args:
            data (numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = data.copy()
        if self._model_missing_values:
            if self.nulls:
                isna = data[:, 1] > 0.5

            data = data[:, 0]

        elif self.nulls:
            isna = np.random.random((len(data),)) < self._null_percentage

        data = pd.Series(data)

        if self.nulls and isna.any():
            data.loc[isna] = np.nan

        return data
