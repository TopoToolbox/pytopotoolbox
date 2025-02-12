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

    # Ensure that the source and target arrays contain valid pixel indices
    assert np.all((0 <= fd.source) & (fd.source < np.prod(dem.shape)))
    assert np.all((0 <= fd.target) & (fd.target < np.prod(dem.shape)))

    # Run flow_accumulation at least once during the tests
    acc = fd.flow_accumulation()

    # Ensure that FlowObject does not modify the original DEM
    assert np.all(dem.z == original_dem)

