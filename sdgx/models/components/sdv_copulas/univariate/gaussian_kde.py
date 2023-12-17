"""GaussianKDE module."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import gaussian_kde

from sdgx.models.components.sdv_copulas import (
    EPSILON,
    random_state,
    store_args,
    validate_random_state,
)
from sdgx.models.components.sdv_copulas.optimize import bisect, chandrupatla
from sdgx.models.components.sdv_copulas.univariate.base import (
    BoundedType,
    ParametricType,
    ScipyModel,
)


class GaussianKDE(ScipyModel):
    """A wrapper for gaussian Kernel density estimation.

    It was implemented in scipy.stats toolbox. gaussian_kde is slower than statsmodels
    but allows more flexibility.

    When a sample_size is provided the fit method will sample the
    data, and mask the real information. Also, ensure the number of
    entries will be always the value of sample_size.

    Args:
        sample_size(int): amount of parameters to sample
    """

    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED
    MODEL_CLASS = gaussian_kde

    @store_args
    def __init__(self, sample_size=None, random_state=None, bw_method=None, weights=None):
        self.random_state = validate_random_state(random_state)
        self._sample_size = sample_size
        self.bw_method = bw_method
        self.weights = weights

    def _get_model(self):
        dataset = self._params["dataset"]
        self._sample_size = self._sample_size or len(dataset)
        return gaussian_kde(dataset, bw_method=self.bw_method, weights=self.weights)

    def _get_bounds(self):
        X = self._params["dataset"]
        lower = np.min(X) - (5 * np.std(X))
        upper = np.max(X) + (5 * np.std(X))

        return lower, upper

    def probability_density(self, X):
        """Compute the probability density for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the probability density will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Probability density values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._model.evaluate(X)

    @random_state
    def sample(self, n_samples=1):
        """Sample values from this model.

        Argument:
            n_samples (int):
                Number of values to sample

        Returns:
            numpy.ndarray:
                Array of shape (n_samples, 1) with values randomly
                sampled from this model distribution.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        return self._model.resample(size=n_samples)[0]

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution value for each point in X.

        Arguments:
            X (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1).

        Returns:
            numpy.ndarray:
                Cumulative distribution values for points in X.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()
        X = np.array(X)
        stdev = np.sqrt(self._model.covariance[0, 0])
        lower = ndtr((self._get_bounds()[0] - self._model.dataset) / stdev)[0]
        uppers = ndtr((X[:, None] - self._model.dataset) / stdev)
        return (uppers - lower).dot(self._model.weights)

    def percent_point(self, U, method="chandrupatla"):
        """Compute the inverse cumulative distribution value for each point in U.

        Arguments:
            U (numpy.ndarray):
                Values for which the cumulative distribution will be computed.
                It must have shape (n, 1) and values must be in [0,1].
            method (str):
                Whether to use the `chandrupatla` or `bisect` solver.

        Returns:
            numpy.ndarray:
                Inverse cumulative distribution values for points in U.

        Raises:
            NotFittedError:
                if the model is not fitted.
        """
        self.check_fit()

        if len(U.shape) > 1:
            raise ValueError(f"Expected 1d array, got {(U, )}.")

        if np.any(U > 1.0) or np.any(U < 0.0):
            raise ValueError("Expected values in range [0.0, 1.0].")

        is_one = U >= 1.0 - EPSILON
        is_zero = U <= EPSILON
        is_valid = ~(is_zero | is_one)

        lower, upper = self._get_bounds()

        def _f(X):
            return self.cumulative_distribution(X) - U[is_valid]

        X = np.zeros(U.shape)
        X[is_one] = float("inf")
        X[is_zero] = float("-inf")
        if is_valid.any():
            lower = np.full(U[is_valid].shape, lower)
            upper = np.full(U[is_valid].shape, upper)
            if method == "bisect":
                X[is_valid] = bisect(_f, lower, upper)
            else:
                X[is_valid] = chandrupatla(_f, lower, upper)

        return X

    def _fit_constant(self, X):
        sample_size = self._sample_size or len(X)
        constant = np.unique(X)[0]
        self._params = {
            "dataset": [constant] * sample_size,
        }

    def _fit(self, X):
        if self._sample_size:
            X = gaussian_kde(X, bw_method=self.bw_method, weights=self.weights).resample(
                self._sample_size
            )
        self._params = {"dataset": X.tolist()}
        self._model = self._get_model()

    def _is_constant(self):
        return len(np.unique(self._params["dataset"])) == 1

    def _extract_constant(self):
        return self._params["dataset"][0]

    def _set_params(self, params):
        """Set the parameters of this univariate.

        Args:
            params (dict):
                Parameters to recreate this instance.
        """
        self._params = params.copy()
        if self._is_constant():
            constant = self._extract_constant()
            self._set_constant_value(constant)
        else:
            self._model = self._get_model()
