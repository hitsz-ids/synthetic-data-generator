"""Transformer for boolean data."""

import numpy as np
import pandas as pd

from sdgx.models.components.sdv_rdt.transformers.base import BaseTransformer
from sdgx.models.components.sdv_rdt.transformers.null import NullTransformer


class BinaryEncoder(BaseTransformer):
    """Transformer for boolean data.

    This transformer replaces boolean values with their integer representation
    transformed to float.

    Null values are replaced using a ``NullTransformer``.

    Args:
        missing_value_replacement (object or None):
            Indicate what to do with the null values. If an object is given, replace them
            with the given value. If the string ``'mode'`` is given, replace them with the
            most common value. If ``None`` is given, do not replace them.
            Defaults to ``None``.
        model_missing_values (bool):
            Whether to create a new column to indicate which values were null or not. The column
            will be created only if there are null values. If ``True``, create the new column if
            there are null values. If ``False``, do not create the new column even if there
            are null values. Defaults to ``False``.
    """

    INPUT_SDTYPE = "boolean"
    DETERMINISTIC_TRANSFORM = True
    DETERMINISTIC_REVERSE = True

    null_transformer = None

    def __init__(self, missing_value_replacement=None, model_missing_values=False):
        self.missing_value_replacement = missing_value_replacement
        self.model_missing_values = model_missing_values

    def get_output_sdtypes(self):
        """Return the output sdtypes returned by this transformer.

        Returns:
            dict:
                Mapping from the transformed column names to the produced sdtypes.
        """
        output_sdtypes = {
            "value": "float",
        }
        if self.null_transformer and self.null_transformer.models_missing_values():
            output_sdtypes["is_null"] = "float"

        return self._add_prefix(output_sdtypes)

    def _fit(self, data):
        """Fit the transformer to the data.

        Args:
            data (pandas.Series):
                Data to fit to.
        """
        self.null_transformer = NullTransformer(
            self.missing_value_replacement, self.model_missing_values
        )
        self.null_transformer.fit(data)

    def _transform(self, data):
        """Transform boolean to float.

        The boolean values will be replaced by the corresponding integer
        representations as float values.

        Args:
            data (pandas.Series):
                Data to transform.

        Returns
            pandas.DataFrame or pandas.Series
        """
        data = pd.to_numeric(data, errors="coerce")
        return self.null_transformer.transform(data).astype(float)

    def _reverse_transform(self, data):
        """Transform float values back to the original boolean values.

        Args:
            data (pandas.DataFrame or pandas.Series):
                Data to revert.

        Returns:
            pandas.Series:
                Reverted data.
        """
        if not isinstance(data, np.ndarray):
            data = data.to_numpy()

        if self.missing_value_replacement is not None:
            data = self.null_transformer.reverse_transform(data)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = data[:, 0]

            data = pd.Series(data)

        isna = data.isna()
        data = np.round(data).clip(0, 1).astype("boolean").astype("object")
        data[isna] = np.nan

        return data
