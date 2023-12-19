import numpy as np
import pytest

from sdgx.models.components.optimize.ndarray_loader import NDArrayLoader


@pytest.fixture
def ndarray_loader(tmp_path, ndarray_list):
    cache_dir = tmp_path / "ndarrycache"
    loader = NDArrayLoader(cache_root=cache_dir)
    for ndarray in ndarray_list:
        loader.store(ndarray)
    yield loader
    loader.cleanup()


@pytest.fixture
def ndarray_list():
    """
    1, 4, 7
    2, 5, 8
    3, 6, 9
    """
    yield [
        np.array([[1], [2], [3]]),
        np.array([[4], [5], [6]]),
        np.array([[7], [8], [9]]),
    ]


def test_ndarray_loader_function(ndarray_loader: NDArrayLoader, ndarray_list):
    ndarray_all = np.concatenate(ndarray_list, axis=1)

    for i, ndarray in enumerate(ndarray_loader.iter()):
        np.testing.assert_equal(ndarray, ndarray_list[i])
    np.testing.assert_equal(ndarray_loader.get_all(), ndarray_all)

    assert ndarray_loader.shape == ndarray_all.shape


def test_ndarray_loader_slice(ndarray_loader: NDArrayLoader, ndarray_list):
    ndarray_all = np.concatenate(ndarray_list, axis=1)

    np.testing.assert_equal(ndarray_loader[:], ndarray_all[:])
    np.testing.assert_equal(ndarray_loader[::], ndarray_all[::])
    np.testing.assert_equal(ndarray_loader[:, :], ndarray_all[:, :])
    np.testing.assert_equal(ndarray_loader[::, ::], ndarray_all[::, ::])
    np.testing.assert_equal(ndarray_loader[:, 1], ndarray_all[:, 1])
    np.testing.assert_equal(ndarray_loader[1, :], ndarray_all[1, :])

    np.testing.assert_equal(ndarray_loader[1:3], ndarray_all[1:3])
    """
    2, 3
    5, 6
    8, 9
    """

    np.testing.assert_equal(ndarray_loader[1:3, 1], ndarray_all[1:3, 1])
    """
    5
    6
    """

    np.testing.assert_equal(ndarray_loader[1:3, 1:3], ndarray_all[1:3, 1:3])
    """
    5, 6
    8, 9
    """


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
