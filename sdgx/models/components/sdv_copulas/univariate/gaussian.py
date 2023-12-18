"""GaussianUnivariate module."""

import numpy as np
from scipy.stats import norm

from sdgx.models.components.sdv_copulas.univariate.base import (
    BoundedType,
    ParametricType,
    ScipyModel,
)


class GaussianUnivariate(ScipyModel):
    """Gaussian univariate model."""

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = norm

    def _fit_constant(self, X):
        self._params = {"loc": np.unique(X)[0], "scale": 0}

    def _fit(self, X):
        self._params = {"loc": np.mean(X), "scale": np.std(X)}

    def _is_constant(self):
        return self._params["scale"] == 0

    def _extract_constant(self):
        return self._params["loc"]
