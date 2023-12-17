"""Sample datasets for the Copulas library."""

import numpy as np
import pandas as pd
from scipy import stats

from sdgx.models.components.sdv_copulas import set_random_state, validate_random_state


def _dummy_fn(state):
    pass


def sample_bivariate_age_income(size=1000, seed=42):
    """Sample from a bivariate toy dataset.

    This dataset contains two columns which correspond to the simulated age and
    income which are positively correlated with outliers.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.DataFrame:
            DataFrame with two columns, ``age`` and ``income``.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        age = stats.beta.rvs(a=2.0, b=6.0, loc=18, scale=100, size=size)
        income = np.log(age) * 100
        income += np.random.normal(loc=np.log(age) / 100, scale=10, size=size)
        income[np.random.randint(0, 10, size=size) == 0] /= 1000

    return pd.DataFrame({"age": age, "income": income})


def sample_trivariate_xyz(size=1000, seed=42):
    """Sample from three dimensional toy dataset.

    The output is a DataFrame containing three columns:

    * ``x``: Beta distribution with a=0.1 and b=0.1
    * ``y``: Beta distribution with a=0.1 and b=0.5
    * ``z``: Normal distribution + 10 times ``y``

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.DataFrame:
            DataFrame with three columns, ``x``, ``y`` and ``z``.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        x = stats.beta.rvs(a=0.1, b=0.1, size=size)
        y = stats.beta.rvs(a=0.1, b=0.5, size=size)
        return pd.DataFrame({"x": x, "y": y, "z": np.random.normal(size=size) + y * 10})


def sample_univariate_bernoulli(size=1000, seed=42):
    """Sample from a Bernoulli distribution with p=0.3.

    The distribution is built by sampling a uniform random and then setting
    0 or 1 depending on whether the value is above or below 0.3.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.random.random(size=size) < 0.3).astype(float)


def sample_univariate_bimodal(size=1000, seed=42):
    """Sample from a bimodal distribution which mixes two Gaussians at 0.0 and 10.0 with stdev=1.

    The distribution is built by sampling a standard normal and a normal with mean ``10``
    and then selecting one or the other based on a bernoulli distribution.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        bernoulli = sample_univariate_bernoulli(size, seed)
        mode1 = np.random.normal(size=size) * bernoulli
        mode2 = np.random.normal(size=size, loc=10) * (1.0 - bernoulli)

        return pd.Series(mode1 + mode2)


def sample_univariate_uniform(size=1000, seed=42):
    """Sample from a uniform distribution in [-1.0, 3.0].

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(4.0 * np.random.random(size=size) - 1.0)


def sample_univariate_normal(size=1000, seed=42):
    """Sample from a normal distribution with mean 1 and stdev 1.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.random.normal(size=size, loc=1.0))


def sample_univariate_degenerate(size=1000, seed=42):
    """Sample from a degenerate distribution that only takes one random value.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.full(size, np.random.random()))


def sample_univariate_exponential(size=1000, seed=42):
    """Sample from an exponential distribution at 3.0 with rate 1.0.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(np.random.exponential(size=size) + 3.0)


def sample_univariate_beta(size=1000, seed=42):
    """Sample from a beta distribution with a=3 and b=1 and loc=4.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.Series:
            Series with the sampled values.
    """
    with set_random_state(validate_random_state(seed), _dummy_fn):
        return pd.Series(stats.beta.rvs(a=3, b=1, loc=4, size=size))


def sample_univariates(size=1000, seed=42):
    """Sample from a list of univariate distributions.

    Args:
        size (int):
            Amount of samples to generate. Defaults to 1000.
        seed (int):
            Random seed to use. Defaults to 42.

    Returns:
        pandas.DataFrame:
            DataFrame with the sampled distributions.
    """
    return pd.DataFrame(
        {
            "bernoulli": sample_univariate_bernoulli(size, seed),
            "bimodal": sample_univariate_bimodal(size, seed),
            "uniform": sample_univariate_uniform(size, seed),
            "normal": sample_univariate_normal(size, seed),
            "degenerate": sample_univariate_degenerate(size, seed),
            "exponential": sample_univariate_exponential(size, seed),
            "beta": sample_univariate_beta(size, seed),
        }
    )
