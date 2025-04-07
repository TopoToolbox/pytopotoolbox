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

    assert isinstance(z, topo.GridObject)

    # ezgetnal is idempotent
    assert np.array_equal(z, wide_dem.z)
    assert np.array_equal(z.z, z2)

    # ezgetnal should always return a copy
    assert z is not wide_dem
    assert z2 is not z.z

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

