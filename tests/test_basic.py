import numpy as np
import pytest

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


def test_wrong_shape(small_array):
    stats = npo.NpOnlineStats(small_array)
    with pytest.raises(ValueError, match="Array shape does not match"):
        stats.add(np.ones((2, 2, 3)))


def test_wrong_ndim(small_array):
    stats = npo.NpOnlineStats(small_array)
    with pytest.raises(ValueError, match="Array shape does not match"):
        stats.add(np.ones((4, 2)))
