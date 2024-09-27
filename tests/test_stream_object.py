import pytest
import numpy as np

import topotoolbox as topo


@pytest.fixture
def wide_dem():
    dem = topo.gen_random(rows=128, columns=64)
    flow = topo.FlowObject(dem)
    yield topo.StreamObject(flow)


@pytest.fixture
def tall_dem():
    dem = topo.gen_random(rows=64, columns=128)
    flow = topo.FlowObject(dem)
    yield topo.StreamObject(flow)


def test_init(tall_dem, wide_dem):
    assert (wide_dem.target.size == wide_dem.source.size ==
            wide_dem.direction.size == wide_dem.stream.size)
    assert (tall_dem.target.size == tall_dem.source.size ==
            tall_dem.direction.size == tall_dem.stream.size)

    # Ensure that no index in stream exceeds the max possible index in the grid
    assert np.max(wide_dem.stream) <= wide_dem.shape[0] * wide_dem.shape[1]
    assert np.max(tall_dem.stream) <= tall_dem.shape[0] * tall_dem.shape[1]

    grid_obj = topo.gen_random(rows=64, columns=64)
    flow_obj = topo.FlowObject(grid_obj)

    # The first positional argument has to be a FlowObject.
    with pytest.raises(TypeError):
        topo.StreamObject(1)

    # Units can only be 'pixels','km2','m2' or 'mapunits'
    with pytest.raises(ValueError):
        topo.StreamObject(flow_obj, units=' ')

    # The threshold ndarray should have the same shape as flow_obj.
    with pytest.raises(ValueError):
        topo.StreamObject(flow_obj, threshold=np.zeros(shape=(1, 1)))

    # The threshold GridObject should have the same shape as flow_obj.
    with pytest.raises(ValueError):
        topo.StreamObject(flow_obj, threshold=topo.gen_random(
            rows=1, columns=1))
