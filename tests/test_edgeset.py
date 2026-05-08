import pytest

import numpy as np

import topotoolbox as tt3

rng = np.random.default_rng(210057042403268582692858923958297081890)

@pytest.fixture
def dem():
    sz = (11, 13)
    return rng.random(sz, dtype=np.float32)

def bitmap(z):
    sz = z.shape
    edges = np.zeros(sz, dtype=np.uint8)

    ei = np.array([0, -1, -1, -1, 0, 1, 1, 1])
    ej = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    for (index, x) in np.ndenumerate(z):
        for (n, (ni, nj)) in enumerate(zip(index[0] + ei, index[1] + ej)):
            if ((ni < 0) or (ni >= sz[0]) or (nj < 0) or (nj >= sz[1])):
                continue

            if z[ni, nj] < z[index]:
                edges[index] ^= 1 << n

    return edges

@pytest.fixture
def bitmap1(dem):
    return bitmap(dem)

@pytest.fixture
def bitmap2(dem):
    return bitmap(dem)

def uniform_weights(bitmap):
    np.seterr(divide='ignore')
    c = np.bitwise_count(bitmap)
    w = 1 / c.flatten()
    return np.asarray(np.repeat(w, c.flatten()), np.float32)

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
    assert np.array_equal(scan.flatten()[1:], np.cumsum(np.bitwise_count(bitmap1).flatten())[0:-1])
    assert np.array_equal(np.bitwise_or(bitmap1, bitmap2), bitmap1)

def test_edgeset_tsort(bitmap1):
    weights1 = uniform_weights(bitmap1)
    scan = np.zeros(bitmap1.shape, dtype=np.int64)
    c = tt3._flow.edgeset_scan(scan, bitmap1)

    stream = np.zeros(bitmap1.shape, dtype=np.int64)
    source = np.zeros(c, dtype=np.int64)
    target = np.zeros(c, dtype=np.int64)
    sweight= np.zeros(c, dtype=np.float32)

    stack = np.zeros(bitmap1.shape, dtype=np.int64)
    stackdir = np.zeros(bitmap1.shape, dtype=np.uint8)
    visited = np.zeros(bitmap1.shape, dtype=np.uint8)

    tt3._flow.flow_routing_tsort(stream, source, target, sweight,
                                 stack, stackdir, bitmap1, weights1, scan, visited)

    # If the edge lists are topologically sorted, no pixel will be
    # visited before all of its upstream neighbors are visited
    visited[:] = 0
    for (u, v) in zip(source, target):
        visited[np.unravel_index(u, bitmap1.shape)] += 1

        assert visited[np.unravel_index(v, bitmap1.shape)] == 0

    # Make sure that the weight for each edge is the same as that
    # given by the initial weights array. Because each outgoing edge
    # from a pixel has the same weight, we check the weight of the
    # first outgoing edge of each source. This may not be the same
    # edge, but it should have the same weight.
    for (u, w) in zip(source, sweight):
        assert w == weights1[scan[np.unravel_index(u, scan.shape)]]

def test_flow_routing_d8(dem):
    directions = np.zeros(dem.shape, dtype=np.uint8)
    tt3._flow.flow_routing_d8_directions(directions, dem)

    assert np.all(np.bitwise_count(directions) < 2)

    scan = np.zeros(dem.shape, dtype=np.int64)
    c = tt3._flow.edgeset_scan(scan, directions)

    assert c > 0

    weights = np.zeros(c, dtype=np.float32)
    tt3._flow.flow_routing_d8_weights(weights)

    stream = np.zeros(dem.shape, dtype=np.int64)
    source = np.zeros(c, dtype=np.int64)
    target = np.zeros(c, dtype=np.int64)
    sweight= np.zeros(c, dtype=np.float32)

    stack = np.zeros(dem.shape, dtype=np.int64)
    stackdir = np.zeros(dem.shape, dtype=np.uint8)
    visited = np.zeros(dem.shape, dtype=np.uint8)

    tt3._flow.flow_routing_tsort(stream, source, target, sweight,
                                 stack, stackdir, directions, weights, scan, visited)

    assert np.all(sweight == 1.0)

    visited[:] = 0
    for (u, v) in zip(source, target):
        visited[np.unravel_index(u, dem.shape)] += 1

        assert visited[np.unravel_index(v, dem.shape)] == 0

def test_resolve_flats_lcat(dem):
    dims = (dem.shape[1], dem.shape[0])
    demf = np.zeros(dem.shape, dtype=np.float32)
    bc = np.ones_like(dem, dtype=np.uint8)
    bc[1:-1, 1:-1] = 0

    tt3._grid.fillsinks(demf, dem, bc, dims)

    flats = np.zeros_like(dem, dtype=np.int32)
    tt3._grid.identifyflats(flats, demf, dims)

    costs = np.zeros_like(dem, dtype=np.float32)
    conncomps = np.zeros_like(dem, dtype=np.int64)
    tt3._grid.gwdt_computecosts(costs, conncomps, flats, dem, demf, dims)

    aux  = np.zeros_like(flats, dtype=np.float32)
    prev = np.zeros_like(flats, dtype=np.int64)
    heap = np.zeros_like(flats, dtype=np.int64)
    back = np.zeros_like(flats, dtype=np.int64)
    tt3._grid.gwdt(aux, prev, costs, flats, heap, back, dims)

    direction = np.zeros(dem.shape, dtype=np.uint8)
    tt3._flow.flow_routing_d8_directions(direction, demf)

    scan = np.zeros(dem.shape, dtype=np.int64)
    c = tt3._flow.edgeset_scan(scan, direction)

    weights = np.zeros(c, dtype=np.float32)
    tt3._flow.flow_routing_d8_weights(weights)

    rdirection = np.zeros(dem.shape, dtype=np.uint8)
    resolved = flats & 1 == 0

    assert not np.all(resolved)

    tt3._flow.resolve_flats_lcat(rdirection, resolved, aux, demf)

    cr = tt3._flow.edgeset_count(rdirection)

    rweights = np.zeros(cr, dtype=np.float32)
    tt3._flow.resolve_flats_lcat_weights(rweights)

    c2 = tt3._flow.edgeset_count_merged(rdirection, direction)

    mweights = np.zeros(c2, dtype=np.float32)
    mscan = np.zeros(dem.shape, dtype=np.int64)
    tt3._flow.edgeset_merge(mweights, mscan, direction, weights, rdirection, rweights)

    stream = np.zeros(dem.shape, dtype=np.int64)
    source = np.zeros(c2, dtype=np.int64)
    target = np.zeros(c2, dtype=np.int64)
    sweight = np.zeros(c2, dtype=np.float32)

    stack = np.zeros(dem.shape, dtype=np.int64)
    stackdir = np.zeros(dem.shape, dtype=np.uint8)
    visited = np.zeros(dem.shape, dtype=np.uint8)

    tt3._flow.flow_routing_tsort(stream, source, target,
                                 sweight, stack, stackdir,
                                 direction, mweights, mscan, visited)

    assert np.all(sweight == 1.0)

    visited[:] = 0
    for (u, v) in zip(source, target):
        visited[np.unravel_index(u, dem.shape)] += 1

        assert visited[np.unravel_index(v, dem.shape)] == 0

    # All pixels with no outgoing edges should be on the boundary
    assert np.all(bc[visited == 0] == 1)
