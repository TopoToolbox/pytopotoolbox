import copy
import pytest

import numpy as np
import topotoolbox as topo

@pytest.fixture
def wide_dem():
    return topo.gen_random(rows=64, columns=128, seed=12)

def test_flowobject(wide_dem):
    dem = wide_dem
    original_dem = dem.z.copy()
    fd = topo.FlowObject(dem);

    assert topo.validate_alignment(dem, fd)

    # Ensure that the source and target arrays contain valid pixel indices
    assert np.all((0 <= fd.source) & (fd.source < np.prod(dem.shape)))
    assert np.all((0 <= fd.target) & (fd.target < np.prod(dem.shape)))

    # Run flow_accumulation at least once during the tests
    acc = fd.flow_accumulation()

    assert topo.validate_alignment(fd, acc)

    # Ensure that FlowObject does not modify the original DEM
    assert np.all(dem.z == original_dem)

def test_ezgetnal(wide_dem):
    fd = topo.FlowObject(wide_dem)

    z = fd.ezgetnal(wide_dem)
    z2 = fd.ezgetnal(z.z)
    z3 = fd.ezgetnal(wide_dem, dtype=np.float64)

    assert isinstance(z, topo.GridObject)
    assert isinstance(z2, np.ndarray)
    assert isinstance(z3, topo.GridObject)

    # ezgetnal is idempotent
    assert np.array_equal(z, wide_dem)
    assert np.array_equal(z.z, z2)
    assert np.array_equal(z3, wide_dem)

    # ezgetnal should always return a copy
    assert z is not wide_dem
    assert z2 is not z.z
    assert z3 is not wide_dem

    # ezgetnal with dtype should return array of that dtype
    assert z3.z.dtype is np.dtype(np.float64)

def test_flowpathextract(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)

    assert topo.validate_alignment(wide_dem, fd)
    assert topo.validate_alignment(wide_dem, s)

    ch = s.streampoi('channelheads')

    s2 = topo.StreamObject(fd, channelheads=s.stream[ch][0:1])

    assert topo.validate_alignment(wide_dem, s2)

    idxs = fd.flowpathextract(s.stream[ch][0])
    assert np.array_equal(s2.stream, idxs)


def test_imposemin(wide_dem):
    original_dem = wide_dem.z.copy()
    fd = topo.FlowObject(wide_dem)

    for minimum_slope in [0.0,0.001,0.01,0.1]:
        min_dem = topo.imposemin(fd, wide_dem, minimum_slope)

        # The carved dem should not be above the original
        assert np.all(min_dem.z <= wide_dem)

        # The gradient along the flow network should be greater than or
        # equal to the defined slope within some numerical error
        g = (min_dem.z[np.unravel_index(fd.source,fd.shape,order='F')] -
             min_dem.z[np.unravel_index(fd.target,fd.shape,order='F')])/fd.distance()
        assert np.all(g >= minimum_slope - 1e-6)

        # imposemin should not modify the original array
        assert np.array_equal(original_dem, wide_dem.z)

def test_imposemin_f64(wide_dem):
    original_dem = np.array(wide_dem, dtype=np.float64)

    fd = topo.FlowObject(wide_dem)

    z = np.array(wide_dem, dtype=np.float64)

    # The np.float64 type is not supported
    with pytest.raises(ValueError):
        min_dem = topo.imposemin(fd, z, 0.001)
