"""Clayton module."""

import numpy as np

from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate, CopulaTypes
from sdgx.models.components.sdv_copulas.bivariate.utils import split_matrix


class Clayton(Bivariate):
    """Class for clayton copula model."""

    copula_type = CopulaTypes.CLAYTON
    theta_interval = [0, float("inf")]
    invalid_thetas = []

    def generator(self, t):
        r"""Compute the generator function for Clayton copula family.

        The generator is a function
        :math:`\psi: [0,1]\times\Theta \rightarrow [0, \infty)`  # noqa: JS101

        that given an Archimedian copula fulfills:
        .. math:: C(u,v) = \psi^{-1}(\psi(u) + \psi(v))

        Args:
            t (numpy.ndarray)

        Returns:
            numpy.ndarray

        """
        self.check_fit()

        return (1.0 / self.theta) * (np.power(t, -self.theta) - 1)

    def probability_density(self, X):
        r"""Compute probability density function for given copula family.

        The probability density(PDF) for the Clayton family of copulas correspond to the formula:

        .. math:: c(U,V) = \frac{\partial^2}{\partial v \partial u}C(u,v) =
            (\theta + 1)(uv)^{-\theta-1}(u^{-\theta} +
            v^{-\theta} - 1)^{-\frac{2\theta + 1}{\theta}}

        Args:
            X (numpy.ndarray)

        Returns:
            numpy.ndarray: Probability density for the input values.

        """
        self.check_fit()

        U, V = split_matrix(X)

        a = (self.theta + 1) * np.power(U * V, -(self.theta + 1))
        b = np.power(U, -self.theta) + np.power(V, -self.theta) - 1
        c = -(2 * self.theta + 1) / self.theta
        return a * np.power(b, c)

    def cumulative_distribution(self, X):
        """Compute the cumulative distribution function for the clayton copula.

        The cumulative density(cdf), or distribution function for the Clayton family of copulas
        correspond to the formula:

        .. math:: C(u,v) = (u^{-θ} + v^{-θ} - 1)^{-1/θ}

        Args:
            X (numpy.ndarray)

        Returns:
            numpy.ndarray: cumulative probability.

        """
        self.check_fit()

        U, V = split_matrix(X)

        if (V == 0).all() or (U == 0).all():
            return np.zeros(V.shape[0])

        else:
            cdfs = [
                (
                    np.power(
                        np.power(U[i], -self.theta) + np.power(V[i], -self.theta) - 1,
                        -1.0 / self.theta,
                    )
                    if (U[i] > 0 and V[i] > 0)
                    else 0
                )
                for i in range(len(U))
            ]

            return np.array(cdfs)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y (numpy.ndarray): Value of :math:`C(u|v)`.
            v (numpy.ndarray): given value of v.
        """
        self.check_fit()

        if self.theta < 0:
            return V

        else:
            a = np.power(y, self.theta / (-1 - self.theta))
            b = np.power(V, self.theta)

            # If b == 0, self.theta tends to inf,
            # so the next operation tends to 1
            if (b == 0).all():
                return np.ones(len(V))

            return np.power((a + b - 1) / b, -1 / self.theta)

    def partial_derivative(self, X):
        r"""Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

        .. math:: F(v|u) = \frac{\partial C(u,v)}{\partial u} =
            u^{- \theta - 1}(u^{-\theta} + v^{-\theta} - 1)^{-\frac{\theta+1}{\theta}}

        Args:
            X (np.ndarray)
            y (float)

        Returns:
            numpy.ndarray: Derivatives

        """
        self.check_fit()

        U, V = split_matrix(X)

        A = np.power(V, -self.theta - 1)

        # If theta tends to inf, A tends to inf
        # And the next partial_derivative tends to 0
        if (A == np.inf).any():
            return np.zeros(len(V))

        B = np.power(V, -self.theta) + np.power(U, -self.theta) - 1
        h = np.power(B, (-1 - self.theta) / self.theta)
        return A * h

    def compute_theta(self):
        r"""Compute theta parameter using Kendall's tau.

        On Clayton copula this is

        .. math:: τ = θ/(θ + 2) \implies θ = 2τ/(1-τ)
        .. math:: θ ∈ (0, ∞)

        On the corner case of :math:`τ = 1`, return infinite.
        """
        if self.tau == 1:
            return np.inf

        return 2 * self.tau / (1 - self.tau)
