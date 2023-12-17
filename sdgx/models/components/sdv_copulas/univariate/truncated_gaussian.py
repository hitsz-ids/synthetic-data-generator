"""TruncatedGaussian module."""

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.stats import truncnorm

from sdgx.models.components.sdv_copulas import (
    EPSILON,
    store_args,
    validate_random_state,
)
from sdgx.models.components.sdv_copulas.univariate.base import (
    BoundedType,
    ParametricType,
    ScipyModel,
)


class TruncatedGaussian(ScipyModel):
    """Wrapper around scipy.stats.truncnorm.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.BOUNDED
    MODEL_CLASS = truncnorm

    @store_args
    def __init__(self, minimum=None, maximum=None, random_state=None):
        self.random_state = validate_random_state(random_state)
        self.min = minimum
        self.max = maximum

    def _fit_constant(self, X):
        constant = np.unique(X)[0]
        self._params = {"a": constant, "b": constant, "loc": constant, "scale": 0.0}

    def _fit(self, X):
        if self.min is None:
            self.min = X.min() - EPSILON

        if self.max is None:
            self.max = X.max() + EPSILON

        def nnlf(params):
            loc, scale = params
            a = (self.min - loc) / scale
            b = (self.max - loc) / scale
            return truncnorm.nnlf((a, b, loc, scale), X)

        initial_params = X.mean(), X.std()
        optimal = fmin_slsqp(
            nnlf,
            initial_params,
            iprint=False,
            bounds=[(self.min, self.max), (0.0, (self.max - self.min) ** 2)],
        )

        loc, scale = optimal
        a = (self.min - loc) / scale
        b = (self.max - loc) / scale

        self._params = {"a": a, "b": b, "loc": loc, "scale": scale}

    def _is_constant(self):
        return self._params["a"] == self._params["b"]

    def _extract_constant(self):
        return self._params["loc"]
