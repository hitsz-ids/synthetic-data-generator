"""Utilities for bivariate copulas."""

import numpy as np


def split_matrix(X):
    """Split an (n,2) numpy.array into two vectors.

    Args:
        X(numpy.array): Matrix of shape (n,2)

    Returns:
        tuple[numpy.array]: Both of shape (n,)

    """
    if len(X):
        return X[:, 0], X[:, 1]

    return np.array([]), np.array([])
