"""Frank module."""

import sys

import numpy as np
import scipy.integrate as integrate
from scipy.optimize import least_squares

from sdgx.models.components.sdv_copulas import EPSILON
from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate, CopulaTypes
from sdgx.models.components.sdv_copulas.bivariate.utils import split_matrix

MIN_FLOAT_LOG = np.log(sys.float_info.min)
MAX_FLOAT_LOG = np.log(sys.float_info.max)


class Frank(Bivariate):
    """Class for Frank copula model."""

    copula_type = CopulaTypes.FRANK
    theta_interval = [-float("inf"), float("inf")]
    invalid_thetas = [0]

    def generator(self, t):
        """Return the generator function."""
        a = (np.exp(-self.theta * t) - 1) / (np.exp(-self.theta) - 1)
        return -np.log(a)

    def _g(self, z):
        r"""Assist in solving the Frank copula.

        This functions encapsulates :math:`g(z) = e^{-\theta z} - 1` used on Frank copulas.

        Argument:
            z: np.ndarray

        Returns:
            np.ndarray

        """
        return np.exp(-self.theta * z) - 1

    def probability_density(self, X):
        r"""Compute probability density function for given copula family.

        The probability density(PDF) for the Frank family of copulas correspond to the formula:

        .. math:: c(U,V) = \frac{\partial^2 C(u,v)}{\partial v \partial u} =
             \frac{-\theta g(1)(1 + g(u + v))}{(g(u) g(v) + g(1)) ^ 2}

        Where the g function is defined by:

        .. math:: g(x) = e^{-\theta x} - 1

        Args:
            X: `np.ndarray`

        Returns:
            np.array: probability density

        """
        self.check_fit()

        U, V = split_matrix(X)

        if self.theta == 0:
            return U * V

        else:
            num = (-self.theta * self._g(1)) * (1 + self._g(U + V))
            aux = self._g(U) * self._g(V) + self._g(1)
            den = np.power(aux, 2)
            return num / den

    def cumulative_distribution(self, X):
        r"""Compute the cumulative distribution function for the Frank copula.

        The cumulative density(cdf), or distribution function for the Frank family of copulas
        correspond to the formula:

        .. math:: C(u,v) =  −\frac{\ln({\frac{1 + g(u) g(v)}{g(1)}})}{\theta}


        Args:
            X: `np.ndarray`

        Returns:
            np.array: cumulative distribution

        """
        self.check_fit()

        U, V = split_matrix(X)

        num = (np.exp(-self.theta * U) - 1) * (np.exp(-self.theta * V) - 1)
        den = np.exp(-self.theta) - 1

        return -1.0 / self.theta * np.log(1 + num / den)

    def percent_point(self, y, V):
        """Compute the inverse of conditional cumulative distribution :math:`C(u|v)^{-1}`.

        Args:
            y: `np.ndarray` value of :math:`C(u|v)`.
            v: `np.ndarray` given value of v.
        """
        self.check_fit()

        if self.theta == 0:
            return V

        else:
            return super().percent_point(y, V)

    def partial_derivative(self, X):
        r"""Compute partial derivative of cumulative distribution.

        The partial derivative of the copula(CDF) is the conditional CDF.

        .. math:: F(v|u) = \frac{\partial}{\partial u}C(u,v) =
            \frac{g(u)g(v) + g(v)}{g(u)g(v) + g(1)}

        Args:
            X (np.ndarray)
            y (float)

        Returns:
            np.ndarray

        """
        self.check_fit()

        U, V = split_matrix(X)

        if self.theta == 0:
            return V

        else:
            num = self._g(U) * self._g(V) + self._g(U)
            den = self._g(U) * self._g(V) + self._g(1)
            return num / den

    def compute_theta(self):
        r"""Compute theta parameter using Kendall's tau.

        On Frank copula, the relationship between tau and theta is defined by:

        .. math:: \tau = 1 − \frac{4}{\theta} + \frac{4}{\theta^2}\int_0^\theta \!
            \frac{t}{e^t -1} \mathrm{d}t.

        In order to solve it, we can simplify it as

        .. math:: 0 = 1 + \frac{4}{\theta}(D_1(\theta) - 1) - \tau

        where the function D is the Debye function of first order, defined as:

        .. math:: D_1(x) = \frac{1}{x}\int_0^x\frac{t}{e^t -1} \mathrm{d}t.

        """
        result = least_squares(self._tau_to_theta, 1, bounds=(MIN_FLOAT_LOG, MAX_FLOAT_LOG))
        return result.x[0]

    def _tau_to_theta(self, alpha):
        """Relationship between tau and theta as a solvable equation."""

        def debye(t):
            return t / (np.exp(t) - 1)

        debye_value = integrate.quad(debye, EPSILON, alpha)[0] / alpha
        return 4 * (debye_value - 1) / alpha + 1 - self.tau
