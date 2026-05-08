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
def edgeset1(dem):
    directions = bitmap(dem)
    scan = np.zeros_like(directions, dtype=np.int64)
    tt3._flow.edgeset_scan(scan, directions)
    weights = uniform_weights(directions)

    return tt3.EdgeSet(directions, scan, weights)

@pytest.fixture
def edgeset2(dem):
    directions = bitmap(dem)
    scan = np.zeros_like(directions, dtype=np.int64)
    tt3._flow.edgeset_scan(scan, directions)
    weights = uniform_weights(directions)

    return tt3.EdgeSet(directions, scan, weights)

def uniform_weights(bitmap):
    np.seterr(divide='ignore')
    c = np.bitwise_count(bitmap)
    w = 1 / c.flatten()
    return np.asarray(np.repeat(w, c.flatten()), np.float32)

# Numpy has a popcnt primitive that should be identical to
# edgeset_count
def test_edgeset_count(edgeset1):
    c1 = edgeset1.count
    c2 = np.sum(np.bitwise_count(edgeset1.directions))
    assert c1 == c2

# The bitmap scan should be identical to the cumsum of the popcnt
# array
def test_edgeset_scan(edgeset1):
    assert np.array_equal(np.cumsum(np.bitwise_count(edgeset1.directions))[0:-1], edgeset1.scan.flatten()[1:])

def test_edgeset_merge(edgeset1, edgeset2):
    merged_edges = edgeset1.merge(edgeset2)

    assert np.array_equal(merged_edges.scan.flatten()[1:], np.cumsum(np.bitwise_count(merged_edges.directions).flatten())[0:-1])
    assert np.array_equal(np.bitwise_or(edgeset1.directions, edgeset2.directions), merged_edges.directions)

def test_edgeset_tsort(edgeset1):
    (stream, source, target, sweight) = edgeset1.tsort()

    visited = np.zeros_like(edgeset1.directions)

    # If the edge lists are topologically sorted, no pixel will be
    # visited before all of its upstream neighbors are visited
    visited[:] = 0
    for (u, v) in zip(source, target):
        visited[np.unravel_index(u, edgeset1.shape)] += 1

        assert visited[np.unravel_index(v, edgeset1.shape)] == 0

    # Make sure that the weight for each edge is the same as that
    # given by the initial weights array. Because each outgoing edge
    # from a pixel has the same weight, we check the weight of the
    # first outgoing edge of each source. This may not be the same
    # edge, but it should have the same weight.
    assert np.array_equal(sweight, edgeset1.weights[edgeset1.scan[np.unravel_index(source, edgeset1.shape)]])

def test_flow_routing_d8(dem):
    directions = np.zeros(dem.shape, dtype=np.uint8)
    tt3._flow.flow_routing_d8_directions(directions, dem)

    assert np.all(np.bitwise_count(directions) < 2)

    scan = np.zeros(dem.shape, dtype=np.int64)
    c = tt3._flow.edgeset_scan(scan, directions)

    assert c > 0

    weights = np.zeros(c, dtype=np.float32)
    tt3._flow.flow_routing_d8_weights(weights)

    edges = tt3.EdgeSet(directions, scan, weights)

    (stream, source, target, sweight) = edges.tsort()

    assert np.all(sweight == 1.0)

    visited = np.zeros_like(directions)
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

    e1 = tt3.EdgeSet(direction, scan, weights)

    rdirection = np.zeros(dem.shape, dtype=np.uint8)
    resolved = flats & 1 == 0

    assert not np.all(resolved)

    tt3._flow.resolve_flats_lcat(rdirection, resolved, aux, demf)

    rscan = np.zeros(dem.shape, dtype=np.int64)
    cr = tt3._flow.edgeset_scan(rscan, rdirection)

    rweights = np.zeros(cr, dtype=np.float32)
    tt3._flow.resolve_flats_lcat_weights(rweights)

    e2 = tt3.EdgeSet(rdirection, rscan, rweights)

    em = e1.merge(e2)

    (stream, source, target, sweight) = em.tsort()

    assert np.all(sweight == 1.0)

    visited = np.zeros_like(direction)
    for (u, v) in zip(source, target):
        visited[np.unravel_index(u, dem.shape)] += 1

        assert visited[np.unravel_index(v, dem.shape)] == 0

    # All pixels with no outgoing edges should be on the boundary
    assert np.all(bc[visited == 0] == 1)
