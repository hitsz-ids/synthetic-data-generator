"""Transformer for datetime data."""

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype
from pandas.core.tools.datetimes import _guess_datetime_format_for_array

from sdgx.models.components.sdv_rdt.transformers.base import BaseTransformer
from sdgx.models.components.sdv_rdt.transformers.null import NullTransformer


class UnixTimestampEncoder(BaseTransformer):
    """Transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an object is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    """

    INPUT_SDTYPE = "datetime"
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True
    COMPOSITION_IS_IDENTITY = True

    null_transformer = None

    def __init__(
        self, missing_value_replacement=None, model_missing_values=False, datetime_format=None
    ):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values
        self.datetime_format = datetime_format
        self._dtype = None

    def is_composition_identity(self):
        """Return whether composition of transform and reverse transform produces the input data.

        Returns:
            bool:
                Whether or not transforming and then reverse transforming returns the input data.
        """
        if self.null_transformer and not self.null_transformer.models_missing_values():
            return False

        return self.COMPOSITION_IS_IDENTITY

    def get_output_sdtypes(self):
        """Return the output sdtypes supported by the transformer.

        Returns:
            dict:
                Mapping from the transformed column names to supported sdtypes.
        """
        output_sdtypes = {
            "value": "float",
        }
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes["is_null"] = "float"

        return self._add_prefix(output_sdtypes)

    def _convert_to_datetime(self, data):
        if data.dtype == "object":
            try:
                pandas_datetime_format = None
                if self.datetime_format:
                    pandas_datetime_format = self.datetime_format.replace("%-", "%")

                data = pd.to_datetime(data, format=pandas_datetime_format)

            except ValueError as error:
                if "Unknown string format:" in str(error):
                    message = "Data must be of dtype datetime, or castable to datetime."
                    raise TypeError(message) from None

                raise ValueError("Data does not match specified datetime format.") from None

        return data

    def _transform_helper(self, datetimes):
        """Transform datetime values to integer."""
        datetimes = self._convert_to_datetime(datetimes)
        nulls = datetimes.isna()
        integers = pd.to_numeric(datetimes, errors="coerce").to_numpy().astype(np.float64)
        integers[nulls] = np.nan
        transformed = pd.Series(integers)

        return transformed

    def _reverse_transform_helper(self, data):
        """Transform integer values back into datetimes."""
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if self.model_missing_values or self.missing_value_replacement is not None:
            data = self.null_transformer.reverse_transform(data)

        data = np.round(data.astype(np.float64))
        return data

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit the transformer to.
        """
        self._dtype = data.dtype
        if self.datetime_format is None:
            datetime_array = data.astype(str).to_numpy()
            self.datetime_format = _guess_datetime_format_for_array(datetime_array)

        transformed = self._transform_helper(data)
        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.model_missing_values
        )
        self.null_transformer.fit(transformed)

    def _transform(self, data):
        """Transform datetime values to float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns:
            numpy.ndarray
        """
        data = self._transform_helper(data)
        return self.null_transformer.transform(data)

    def _reverse_transform(self, data):
        """Convert float values back to datetimes.

        Args:
            data (pandas.Series or numpy.ndarray):
                Data to transform.

        Returns:
            pandas.Series
        """
        data = self._reverse_transform_helper(data)
        datetime_data = pd.to_datetime(data)
        if not isinstance(datetime_data, pd.Series):
            datetime_data = pd.Series(datetime_data)

        if self.datetime_format:
            if self._dtype == "object":
                datetime_data = datetime_data.dt.strftime(self.datetime_format)
            elif is_datetime64_dtype(self._dtype) and ".%f" not in self.datetime_format:
                datetime_data = pd.to_datetime(datetime_data.dt.strftime(self.datetime_format))

        return datetime_data


class OptimizedTimestampEncoder(UnixTimestampEncoder):
    """Optimized transformer for datetime data.

    This transformer replaces datetime values with an integer timestamp transformed to float.
    It optimizes the output values by finding the smallest time unit that is not zero on
    the training datetimes and dividing the generated numerical values by the value of the next
    smallest time unit. This, apart from reducing the orders of magnitude of the transformed
    values, ensures that reverted values always are zero on the lower time units.

    Null values are replaced using a ``NullTransformer``.

    This class behaves exactly as the ``UnixTimestampEncoder`` except with the optimization.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an object is given, replace them
            with the given value. If the strings ``'mean'`` or ``'mode'`` are given, replace
            them with the corresponding aggregation. If ``None`` is given, do not replace them.
            Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
        datetime_format (str):
            The strftime to use for parsing time. For more information, see
            https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    """

    divider = None

    def __init__(
        self, missing_value_replacement=None, model_missing_values=False, datetime_format=None
    ):
        super().__init__(
            missing_value_replacement=missing_value_replacement,
            model_missing_values=model_missing_values,
            datetime_format=datetime_format,
        )

    def _find_divider(self, transformed):
        self.divider = 1
        multipliers = [10] * 9 + [60, 60, 24]
        for multiplier in multipliers:
            candidate = self.divider * multiplier
            if (transformed % candidate).any():
                break

            self.divider = candidate

    def _transform_helper(self, data):
        """Transform datetime values to integer."""
        data = super()._transform_helper(data)
        self._find_divider(data)
        return data // self.divider

    def _reverse_transform_helper(self, data):
        """Transform integer values back into datetimes."""
        data = super()._reverse_transform_helper(data)
        return data * self.divider
