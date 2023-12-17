"""Utils for the dataset generators."""

import numpy as np


def add_nans(array):
    """Add a random amount of NaN values to the given array.

    Args:
        array (np.array):
            1 dimensional numpy array.

    Returns:
        np.array:
            The same array with some values replaced by NaNs.
    """
    if array.dtype.kind == "i":
        array = array.astype(float)

    length = len(array)
    num_nulls = np.random.randint(1, length)
    nulls = np.random.choice(range(length), num_nulls)
    array[nulls] = np.nan
    return array
