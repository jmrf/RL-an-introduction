import pytest
import numpy as np

from common import incremental_average, incremental_weighted_average


def weighted_average(values, alpha):
    n = len(values)
    return np.sum([
        np.power(1 - alpha, n - i) * v
        for i, v in enumerate(values)
    ])


@pytest.mark.parametrize('values, alpha', [(list(range(100)), 1)])
def test_incremental_average(values, alpha):
    avg = 0
    for i, v in enumerate(values):
        avg = incremental_average(v, avg, i + 1)
    assert avg == np.mean(values)


@pytest.mark.parametrize('values, alpha', [(list(range(100)), 0.1)])
def test_incremental_weighted_average(values, alpha):
    avg = 0
    for i, v in enumerate(values):
        avg = incremental_weighted_average(v, avg, i + 1, alpha)

    assert np.isclose(avg, weighted_average(values, alpha), 1e-5)
