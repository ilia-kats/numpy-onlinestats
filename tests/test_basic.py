import numpy as np

import numpy_onlinestats as npo


def test_digest():
    test = np.asarray(range(8)).reshape((2, 2, 2)).astype(np.float32)
    stats = npo.NpOnlineStats(test)
    for i in range(2, 11):
        stats.add(i * test)

    assert (stats.quantile(0.49) == 5.5 * test).all()
    assert (stats.quantile(0.09) == test).all()
