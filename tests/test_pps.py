import pytest

import numpy as np
import topotoolbox as tt3

@pytest.fixture(name="stream")
def fixture_stream():
    dem = tt3.gen_random(rows=64, columns=128)
    fd = tt3.FlowObject(dem)
    s = tt3.StreamObject(fd)
    return s

def test_pps(stream):
    nal = stream.downstream_distance() > 10
    pps = tt3.PPS.from_nal(stream, nal)

    assert pps.npoints == np.count_nonzero(nal)
