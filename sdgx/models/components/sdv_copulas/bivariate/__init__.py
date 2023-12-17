"""Bivariate copulas."""

import numpy as np
import pandas as pd

from sdgx.models.components.sdv_copulas import EPSILON
from sdgx.models.components.sdv_copulas.bivariate.base import Bivariate, CopulaTypes
from sdgx.models.components.sdv_copulas.bivariate.clayton import Clayton
from sdgx.models.components.sdv_copulas.bivariate.frank import Frank
from sdgx.models.components.sdv_copulas.bivariate.gumbel import Gumbel
from sdgx.models.components.sdv_copulas.bivariate.utils import split_matrix

__all__ = (
    "Bivariate",
    "Clayton",
    "CopulaTypes",
    "Frank",
    "Gumbel",
)


COMPUTE_EMPIRICAL_STEPS = 50


def _compute_empirical(X):
    """Compute empirical distribution.

    Args:
        X(numpy.array): Shape (n,2); Datapoints to compute the empirical(frequentist) copula.

    Return:
        tuple(list):

    """
    z_left = []
    z_right = []
    L = []
    R = []

    U, V = split_matrix(X)
    N = len(U)
    base = np.linspace(EPSILON, 1.0 - EPSILON, COMPUTE_EMPIRICAL_STEPS)
    # See https://github.com/sdv-dev/Copulas/issues/45

    for k in range(COMPUTE_EMPIRICAL_STEPS):
        left = sum(np.logical_and(U <= base[k], V <= base[k])) / N
        right = sum(np.logical_and(U >= base[k], V >= base[k])) / N

        if left > 0:
            z_left.append(base[k])
            L.append(left / base[k] ** 2)

        if right > 0:
            z_right.append(base[k])
            R.append(right / (1 - z_right[k]) ** 2)

    return z_left, L, z_right, R


def _compute_tail(c, z):
    r"""Compute upper concentration function for tail.

    The upper tail concentration function is defined by:

    .. math:: R(z) = \frac{[1 − 2z + C(z, z)]}{(1 − z)^{2}}

    Args:
        c(Iterable): Values of :math:`C(z,z)`.
        z(Iterable): Values for the empirical copula.

    Returns:
        numpy.ndarray

    """
    return (1.0 - 2 * np.asarray(z) + c) / (np.power(1.0 - np.asarray(z), 2))


def _compute_candidates(copulas, left_tail, right_tail):
    """Compute dependencies.

    Args:
        copulas(list[Bivariate]): Fitted instances of bivariate copulas.
        z_left(list):
        z_right(list):

    Returns:
        tuple[list]: Arrays of left and right dependencies for the empirical copula.


    """
    left = []
    right = []

    X_left = np.column_stack((left_tail, left_tail))
    X_right = np.column_stack((right_tail, right_tail))

    for copula in copulas:
        left.append(copula.cumulative_distribution(X_left) / np.power(left_tail, 2))
        right.append(_compute_tail(copula.cumulative_distribution(X_right), right_tail))

    return left, right


def select_copula(X):
    r"""Select best copula function based on likelihood.

    Given out candidate copulas the procedure proposed for selecting the one
    that best fit to a dataset of pairs :math:`\{(u_j, v_j )\}, j=1,2,...n` , is as follows:

    1. Estimate the most likely parameter :math:`\theta` of each copula candidate for the given
       dataset.

    2. Construct :math:`R(z|\theta)`. Calculate the area under the tail for each of the copula
       candidates.

    3. Compare the areas: :math:`a_u` achieved using empirical copula against the ones
       achieved for the copula candidates. Score the outcome of the comparison from 3 (best)
       down to 1 (worst).

    4. Proceed as in steps 2- 3 with the lower tail and function :math:`L`.

    5. Finally the sum of empirical upper and lower tail functions is compared against
       :math:`R + L`. Scores of the three comparisons are summed and the candidate with the
       highest value is selected.

    Args:
        X(np.ndarray): Matrix of shape (n,2).

    Returns:
        copula: Best copula that fits for it.

    """
    frank = Frank()
    frank.fit(X)

    if frank.tau <= 0:
        return frank

    copula_candidates = [frank]

    # append copulas into the candidate list
    for copula_class in [Clayton, Gumbel]:
        try:
            copula = copula_class()
            copula.tau = frank.tau
            copula._compute_theta()
            copula_candidates.append(copula)
        except ValueError:
            pass

    left_tail, empirical_left_aut, right_tail, empirical_right_aut = _compute_empirical(X)
    candidate_left_auts, candidate_right_auts = _compute_candidates(
        copula_candidates, left_tail, right_tail
    )

    empirical_aut = np.concatenate((empirical_left_aut, empirical_right_aut))
    candidate_auts = [
        np.concatenate((left, right))
        for left, right in zip(candidate_left_auts, candidate_right_auts)
    ]

    # compute L2 distance from empirical distribution
    diff_left = [np.sum((empirical_left_aut - left) ** 2) for left in candidate_left_auts]
    diff_right = [np.sum((empirical_right_aut - right) ** 2) for right in candidate_right_auts]
    diff_both = [np.sum((empirical_aut - candidate) ** 2) for candidate in candidate_auts]

    # calcule ranks
    score_left = pd.Series(diff_left).rank(ascending=False)
    score_right = pd.Series(diff_right).rank(ascending=False)
    score_both = pd.Series(diff_both).rank(ascending=False)

    score = score_left + score_right + score_both

    selected_copula = np.argmax(score.to_numpy())
    return copula_candidates[selected_copula]
