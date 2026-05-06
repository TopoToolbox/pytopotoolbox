import pytest

import numpy as np

import topotoolbox as tt3

rng = np.random.default_rng(210057042403268582692858923958297081890)

@pytest.fixture
def bitmap1():
    return rng.integers(0, 255, size=(11, 13), dtype=np.uint8)

@pytest.fixture
def bitmap2():
    return rng.integers(0, 255, size=(11, 13), dtype=np.uint8)

def uniform_weights(bitmap):
    np.seterr(divide='ignore')
    c = np.bitwise_count(bitmap)
    w = 1 / c.flatten()
    return np.repeat(w, c.flatten())

# Numpy has a popcnt primitive that should be identical to
# edgeset_count
def test_edgeset_count(bitmap1):
    c1 = tt3._flow.edgeset_count(bitmap1)
    c2 = np.sum(np.bitwise_count(bitmap1))
    assert c1 == c2

# The bitmap scan should be identical to the cumsum of the popcnt
# array
def test_edgeset_scan(bitmap1):
    scan1 = np.zeros(bitmap1.shape, dtype=np.int64)
    c1 = tt3._flow.edgeset_scan(scan1, bitmap1)

    assert c1 == tt3._flow.edgeset_count(bitmap1)
    assert np.array_equal(np.cumsum(np.bitwise_count(bitmap1))[0:-1], scan1.flatten()[1:])

def test_edgeset_count_merged(bitmap1, bitmap2):
    c1 = tt3._flow.edgeset_count_merged(bitmap1, bitmap2)

    assert c1 == np.sum(np.bitwise_count(np.bitwise_or(bitmap1, bitmap2)))

def test_edgeset_merge(bitmap1, bitmap2):

    weights1 = uniform_weights(bitmap1)
    weights2 = uniform_weights(bitmap2)

    scan = np.zeros(bitmap1.shape, dtype=np.int64)

    c = tt3._flow.edgeset_count_merged(bitmap1, bitmap2)
    weights = np.zeros(c, dtype=float)

    c1 = tt3._flow.edgeset_merge(weights, scan, bitmap1, weights1, bitmap2, weights2)

    assert c == c1
    assert np.array_equal(scan.flatten()[1:], np.cumsum(np.bitwise_count(bitmap1))[0:-1])
    assert np.array_equal(np.bitwise_or(bitmap1, bitmap2), bitmap1)
