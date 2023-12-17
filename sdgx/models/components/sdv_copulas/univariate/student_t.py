"""StudentTUnivariate module."""

from scipy.stats import t

from sdgx.models.components.sdv_copulas.univariate.base import (
    BoundedType,
    ParametricType,
    ScipyModel,
)


class StudentTUnivariate(ScipyModel):
    """Wrapper around scipy.stats.t.

    Documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
    """

    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    MODEL_CLASS = t

    def _fit_constant(self, X):
        self._fit(X)
        self._params["scale"] = 0

    def _fit(self, X):
        dataframe, loc, scale = t.fit(X)
        self._params = {"df": dataframe, "loc": loc, "scale": scale}

    def _is_constant(self):
        return self._params["scale"] == 0

    def _extract_constant(self):
        return self._params["loc"]
