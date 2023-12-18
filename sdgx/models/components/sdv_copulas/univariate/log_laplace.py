"""LogLaplace module."""

import numpy as np
from scipy.stats import loglaplace

from sdgx.models.components.sdv_copulas.univariate.base import (
    BoundedType,
    ParametricType,
    ScipyModel,
)


class LogLaplace(ScipyModel):
    """Wrapper around scipy.stats.loglaplace.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loglaplace.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.SEMI_BOUNDED
    MODEL_CLASS = loglaplace

    def _fit_constant(self, X):
        self._params = {
            "c": 2.0,
            "loc": np.unique(X)[0],
            "scale": 0.0,
        }

    def _fit(self, X):
        c, loc, scale = loglaplace.fit(X)
        self._params = {
            "c": c,
            "loc": loc,
            "scale": scale,
        }

    def _is_constant(self):
        return self._params["scale"] == 0

    def _extract_constant(self):
        return self._params["loc"]
