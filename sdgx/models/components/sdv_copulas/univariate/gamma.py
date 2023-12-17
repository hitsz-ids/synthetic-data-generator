"""GammaUnivariate module."""

import numpy as np
from scipy.stats import gamma

from sdgx.models.components.sdv_copulas.univariate.base import (
    BoundedType,
    ParametricType,
    ScipyModel,
)


class GammaUnivariate(ScipyModel):
    """Wrapper around scipy.stats.gamma.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.SEMI_BOUNDED
    MODEL_CLASS = gamma

    def _fit_constant(self, X):
        self._params = {
            "a": 0.0,
            "loc": np.unique(X)[0],
            "scale": 0.0,
        }

    def _fit(self, X):
        a, loc, scale = gamma.fit(X)
        self._params = {
            "a": a,
            "loc": loc,
            "scale": scale,
        }

    def _is_constant(self):
        return self._params["scale"] == 0

    def _extract_constant(self):
        return self._params["loc"]
