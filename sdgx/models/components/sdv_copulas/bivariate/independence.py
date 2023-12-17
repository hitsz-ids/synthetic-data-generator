"""Independence module."""

import numpy as np

from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate, CopulaTypes
from sdgx.models.components.sdv_copulas.bivariate.utils import split_matrix


class Independence(Bivariate):
    """This class represent the copula for two independent variables."""

    copula_type = CopulaTypes.INDEPENDENCE

    def fit(self, X):
        """Fit the copula to the given data.

        Args:
            X (numpy.array): Probabilites in a matrix shaped (n, 2)

        Returns:
            None

        """

    def generator(self, t):
        """Compute the generator function for the Copula.

        The generator function is a function f(t), such that an archimedian copula can be
        defined as

        C(u1, ..., uN) = f(f^-1(u1), ..., f^-1(uN)).

        Args:
            t(numpy.array)

        Returns:
            np.array

        """
        return np.log(t)

    def probability_density(self, X):
        """Compute the probability density for the independence copula."""
        return np.all((0.0 <= X) & (X <= 1.0), axis=1).astype(float)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution of the independence bivariate copula is the product.

        Args:
            X(numpy.array): Matrix of shape (n,2), whose values are in [0, 1]

        Returns:
            numpy.array: Cumulative distribution values of given input.

        """
        U, V = split_matrix(X)
        return U * V

    def partial_derivative(self, X):
        """Compute the conditional probability of one event conditiones to the other.

        In the case of the independence copula, due to C(u,v) = u*v, we have that
        F(u|v) = dC/du = v.

        Args:
            X()

        """
        _, V = split_matrix(X)
        return V

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`F(u|v)^-1`.

        Args:
            y: `np.ndarray` value of :math:`F(u|v)`.
            v: `np.ndarray` given value of v.

        """
        self.check_fit()
        return y
