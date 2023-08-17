import numpy as np
import pytest
from scipy.stats import kurtosis, skew

import numpy_onlinestats as npo


@pytest.fixture
def small_array():
    return np.asarray(range(8)).reshape((2, 2, 2)).astype(np.float32)


def test_digest(small_array):
    stats = npo.NpOnlineStats(small_array)
    for i in range(2, 11):
        stats.add(i * small_array)

    assert (stats.quantile(0.49) == 5.5 * small_array).all()
    assert (stats.quantile(0.09) == small_array).all()
    assert (stats.min() == small_array).all()
    assert (stats.max() == 10 * small_array).all()
    assert stats.nacc == 10


def test_reset(small_array):
    stats = npo.NpOnlineStats(small_array)
    for i in range(2, 11):
        stats.add(i * small_array)

    stats.reset()
    assert stats.nacc == 0
    with pytest.raises(IndexError):
        stats.quantile(0.5)
        stats.mean()

    for i in range(1, 11):
        stats.add(i * small_array)
    assert (stats.quantile(0.49) == 5.5 * small_array).all()


def test_scalar():
    stats = npo.NpOnlineStats(np.asarray(5))
    repr(stats)
    assert stats.mean() == 5


def test_statistics():
    arrays = []
    stats = npo.NpOnlineStats()
    for _i in range(100):
        arr = np.random.uniform(size=(5, 3, 7))
        stats.add(arr)
        arrays.append(arr)
    arrays = np.stack(arrays, axis=0)

    assert np.allclose(stats.mean(), arrays.mean(0))
    assert np.allclose(stats.var(), arrays.var(0), rtol=1e-1)
    assert np.allclose(stats.std(), arrays.std(0), rtol=1e-1)
    assert np.allclose(stats.skewness(), skew(arrays, 0))
    assert np.allclose(stats.kurtosis(), kurtosis(arrays, 0))


def test_wrong_shape(small_array):
    stats = npo.NpOnlineStats(small_array)
    with pytest.raises(ValueError, match="Array shape does not match"):
        stats.add(np.ones((2, 2, 3)))


def test_wrong_ndim(small_array):
    stats = npo.NpOnlineStats(small_array)
    with pytest.raises(ValueError, match="Array shape does not match"):
        stats.add(np.ones((4, 2)))


def test_constructor_arg(small_array):
    npo.NpOnlineStats(small_array, 10)


def test_noarrays():
    stats = npo.NpOnlineStats()
    with pytest.raises(IndexError):
        stats.quantile(0.5)
        stats.mean()
