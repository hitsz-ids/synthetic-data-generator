"""Gumbel module."""

import numpy as np

from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate, CopulaTypes
from sdgx.models.components.sdv_copulas.bivariate.utils import split_matrix


class Gumbel(Bivariate):
    """Class for clayton copula model."""

    copula_type = CopulaTypes.GUMBEL
    theta_interval = [1, float("inf")]
    invalid_thetas = []

    def generator(self, t):
        """Return the generator function."""
        return np.power(-np.log(t), self.theta)

    def probability_density(self, X):
        r"""Compute probability density function for given copula family.

        The probability density(PDF) for the Gumbel family of copulas correspond to the formula:

        .. math::

            \begin{align}
                c(U,V)
                    &= \frac{\partial^2 C(u,v)}{\partial v \partial u}
                    &= \frac{C(u,v)}{uv} \frac{((-\ln u)^{\theta}  # noqa: JS101
                    + (-\ln v)^{\theta})^{\frac{2}  # noqa: JS101
                {\theta} - 2 }}{(\ln u \ln v)^{1 - \theta}}  # noqa: JS101
                ( 1 + (\theta-1) \big((-\ln u)^\theta
                + (-\ln v)^\theta\big)^{-1/\theta})
            \end{align}

        Args:
            X (numpy.ndarray)

        Returns:
            numpy.ndarray

        """
        self.check_fit()

        U, V = split_matrix(X)

        if self.theta == 1:
            return U * V

        else:
            a = np.power(U * V, -1)
            tmp = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            b = np.power(tmp, -2 + 2.0 / self.theta)
            c = np.power(np.log(U) * np.log(V), self.theta - 1)
            d = 1 + (self.theta - 1) * np.power(tmp, -1.0 / self.theta)
            return self.cumulative_distribution(X) * a * b * c * d

    def cumulative_distribution(self, X):
        r"""Compute the cumulative distribution function for the Gumbel copula.

        The cumulative density(cdf), or distribution function for the Gumbel family of copulas
        correspond to the formula:

        .. math:: C(u,v) = e^{-((-\ln u)^{\theta} + (-\ln v)^{\theta})^{\frac{1}{\theta}}}

        Args:
            X (np.ndarray)

        Returns:
            np.ndarray: cumulative probability for the given datapoints, cdf(X).

        """
        self.check_fit()

        U, V = split_matrix(X)

        if self.theta == 1:
            return U * V

        else:
            h = np.power(-np.log(U), self.theta) + np.power(-np.log(V), self.theta)
            h = -np.power(h, 1.0 / self.theta)
            cdfs = np.exp(h)
            return cdfs

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y (np.ndarray): value of :math:`C(u|v)`.
            v (np.ndarray): given value of v.

        """
        self.check_fit()

        if self.theta == 1:
            return y

        else:
            return super().percent_point(y, V)

    def partial_derivative(self, X):
        r"""Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

        .. math:: F(v|u) = \frac{\partial C(u,v)}{\partial u} =
            C(u,v)\frac{((-\ln u)^{\theta} + (-\ln v)^{\theta})^{\frac{1}{\theta} - 1}}
            {\theta(- \ln u)^{1 -\theta}}

        Args:
            X (np.ndarray)
            y (float)

        Returns:
            numpy.ndarray

        """
        self.check_fit()

        U, V = split_matrix(X)

        if self.theta == 1:
            return V

        else:
            t1 = np.power(-np.log(U), self.theta)
            t2 = np.power(-np.log(V), self.theta)
            p1 = self.cumulative_distribution(X)
            p2 = np.power(t1 + t2, -1 + 1.0 / self.theta)
            p3 = np.power(-np.log(V), self.theta - 1)
            return p1 * p2 * p3 / V

    def compute_theta(self):
        r"""Compute theta parameter using Kendall's tau.

        On Gumbel copula :math:`\tau` is defined as :math:`τ = \frac{θ−1}{θ}`
        that we solve as :math:`θ = \frac{1}{1-τ}`
        """
        if self.tau == 1:
            raise ValueError("Tau value can't be 1")

        return 1 / (1 - self.tau)
