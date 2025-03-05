import pytest
import warnings
import numpy as np

import topotoolbox as topo

def issubgraph(S1 : topo.StreamObject, S2 : topo.StreamObject):
    """Test whether S1 represents a subgraph of S2
    """
    es1 = set(map(tuple,np.stack((S1.stream[S1.source],S1.stream[S1.target]),axis=1)))
    es2 = set(map(tuple,np.stack((S2.stream[S2.source],S2.stream[S2.target]),axis=1)))
    return es1 <= es2

@pytest.fixture
def wide_dem():
    yield topo.gen_random(rows=128, columns=64)

@pytest.fixture
def tall_dem():
    yield topo.gen_random(columns=64, rows=128)

def test_init(tall_dem, wide_dem):
    tall_flow = topo.FlowObject(tall_dem)
    tall_stream = topo.StreamObject(tall_flow)

    wide_flow = topo.FlowObject(wide_dem)
    wide_stream = topo.StreamObject(wide_flow)

    assert (wide_stream.target.size == wide_stream.source.size ==
            wide_stream.direction.size)
    assert (tall_stream.target.size == tall_stream.source.size ==
            tall_stream.direction.size)

    # Ensure that no index in stream exceeds the max possible index in the grid
    assert np.max(wide_stream.stream) <= wide_stream.shape[0] * wide_stream.shape[1]
    assert np.max(tall_stream.stream) <= tall_stream.shape[0] * tall_stream.shape[1]

    tall_acc = tall_flow.flow_accumulation()
    wide_acc = wide_flow.flow_accumulation()

    # Run the chi transforms
    tall_stream.chitransform(tall_acc)
    wide_stream.chitransform(wide_acc)

    tall_trunk = tall_stream.trunk()
    tall_k1    = tall_stream.klargestconncomps(1)
    tall_k1_trunk = tall_k1.trunk()

    assert issubgraph(tall_trunk, tall_stream)
    assert issubgraph(tall_k1, tall_stream)
    assert not issubgraph(tall_trunk, tall_k1)

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

    # An empty stream_pixels should result in all arrays in the StreamObject
    # being empty (len = 0).
    arr = np.zeros_like(grid_obj.z)
    stream_obj = topo.StreamObject(flow_obj, stream_pixels=arr)
    assert stream_obj.stream.size == 0
    assert stream_obj.source.size == 0
    assert stream_obj.target.size == 0
    assert stream_obj.direction.size == 0

    # When calling with a stream_pixels, threshold should be ignored and a
    # warning should be thrown.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Catch all warnings

        stream_obj = topo.StreamObject(
            flow_obj, stream_pixels=arr, threshold=1000)

        # Check that a warning was raised
        assert len(w) > 0
        assert issubclass(w[-1].category, Warning)
        assert "threshold will be ignored" in str(w[-1].message)
