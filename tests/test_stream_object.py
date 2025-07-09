import warnings
import pytest

import numpy as np
import opensimplex

import topotoolbox as topo

def issubgraph(s1  : topo.StreamObject, s2 : topo.StreamObject):
    """Test whether s1 represents a subgraph of s2
    """
    e1 = set(zip(zip(*s1.source_indices), zip(*s1.target_indices)))
    e2 = set(zip(zip(*s2.source_indices), zip(*s2.target_indices)))
    return e1 <= e2

def isequivalent(s1 : topo.StreamObject, s2 : topo.StreamObject):
    v1 = set(zip(*s1.node_indices))
    v2 = set(zip(*s2.node_indices))

    e1 = set(zip(zip(*s1.source_indices),zip(*s1.target_indices)))
    e2 = set(zip(zip(*s2.source_indices),zip(*s2.target_indices)))

    return v1 == v2 and e1 == e2

@pytest.fixture(name="wide_dem")
def fixture_wide_dem():
    yield topo.gen_random(rows=64, columns=128)

@pytest.fixture(name="tall_dem")
def fixture_tall_dem():
    yield topo.gen_random(rows=128, columns=64)

@pytest.fixture
def order_dems():
    opensimplex.seed(12)

    x = np.arange(0, 128)
    y = np.arange(0, 256)

    cdem = topo.GridObject()
    cdem.z = np.array(
        64 * (opensimplex.noise2array(x/13, y/13) + 1), dtype=np.float32)
    cdem.cellsize = 13.0

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0

    return [cdem, fdem]

def test_constructors():
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

def test_streamobject_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)

    assert isequivalent(cs, fs)

def test_streamobject_streampixels_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    ca = cfd.flow_accumulation()
    fa = ffd.flow_accumulation()

    csp = ca.z > 100
    fsp = fa.z > 100

    cs = topo.StreamObject(cfd, stream_pixels=csp)
    fs = topo.StreamObject(ffd, stream_pixels=fsp)

    assert isequivalent(cs, fs)


def test_streamobject_sizes(tall_dem, wide_dem):
    tall_flow = topo.FlowObject(tall_dem)
    tall_stream = topo.StreamObject(tall_flow)

    assert topo.validate_alignment(tall_dem, tall_flow)
    assert topo.validate_alignment(tall_dem, tall_stream)

    wide_flow = topo.FlowObject(wide_dem)
    wide_stream = topo.StreamObject(wide_flow)

    assert topo.validate_alignment(wide_dem, wide_flow)
    assert topo.validate_alignment(wide_dem, wide_stream)

    assert not topo.validate_alignment(wide_dem, tall_flow)
    assert not topo.validate_alignment(wide_dem, tall_stream)

    assert (wide_stream.target.size == wide_stream.source.size ==
            wide_stream.direction.size)
    assert (tall_stream.target.size == tall_stream.source.size ==
            tall_stream.direction.size)

    # Ensure that no index in stream exceeds the max possible index in the grid
    assert np.max(wide_stream.stream) <= wide_stream.shape[0] * wide_stream.shape[1]
    assert np.max(tall_stream.stream) <= tall_stream.shape[0] * tall_stream.shape[1]


def test_distance_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)
    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)

    cd = cs.distance()
    fd = fs.distance()

    cdg = np.zeros(cfd.shape)
    fdg = np.zeros(ffd.shape)

    cdg[cs.source_indices] = cd
    fdg[fs.source_indices] = fd

    assert np.array_equal(cdg, fdg)


def test_downstream_distance_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)
    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)

    cd = cs.downstream_distance()
    fd = fs.downstream_distance()

    cdg = np.zeros(cfd.shape)
    fdg = np.zeros(ffd.shape)

    cdg[cs.node_indices] = cd
    fdg[fs.node_indices] = fd

    assert np.array_equal(cdg, fdg)

def test_run_chitransform(tall_dem, wide_dem):

    tall_flow = topo.FlowObject(tall_dem)
    tall_stream = topo.StreamObject(tall_flow)

    wide_flow = topo.FlowObject(wide_dem)
    wide_stream = topo.StreamObject(wide_flow)

    tall_acc = tall_flow.flow_accumulation()
    wide_acc = wide_flow.flow_accumulation()

    tall_stream.chitransform(tall_acc)
    wide_stream.chitransform(wide_acc)


def test_chitransform_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)
    ca = cfd.flow_accumulation()

    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)
    fa = cfd.flow_accumulation()

    cchi = cs.chitransform(ca)
    fchi = fs.chitransform(fa)

    cchimap = np.zeros(cs.shape)
    fchimap = np.zeros(fs.shape)

    cchimap[cs.node_indices] = cchi
    fchimap[fs.node_indices] = fchi

    assert np.array_equal(cchimap, fchimap)


def test_stream_subgraphs(tall_dem, wide_dem):
    tall_flow = topo.FlowObject(tall_dem)
    tall_stream = topo.StreamObject(tall_flow)

    tall_trunk = tall_stream.trunk()
    tall_k1    = tall_stream.klargestconncomps(1)
    tall_k1_trunk = tall_k1.trunk()

    assert issubgraph(tall_trunk, tall_stream)
    assert issubgraph(tall_k1, tall_stream)
    assert issubgraph(tall_k1_trunk, tall_stream)
    assert issubgraph(tall_k1_trunk, tall_k1)
    assert not issubgraph(tall_trunk, tall_k1)

    assert topo.validate_alignment(tall_trunk, tall_stream)
    assert topo.validate_alignment(tall_k1, tall_stream)
    assert topo.validate_alignment(tall_k1_trunk, tall_stream)

    wide_flow = topo.FlowObject(wide_dem)
    wide_stream = topo.StreamObject(wide_flow)

    wide_trunk = wide_stream.trunk()
    wide_k1    = wide_stream.klargestconncomps(1)
    wide_k1_trunk = wide_k1.trunk()

    assert issubgraph(wide_trunk, wide_stream)
    assert issubgraph(wide_k1, wide_stream)
    assert issubgraph(wide_k1_trunk, wide_stream)
    assert issubgraph(wide_k1_trunk, wide_k1)
    assert not issubgraph(wide_trunk, wide_k1)

    assert topo.validate_alignment(wide_trunk, wide_stream)
    assert topo.validate_alignment(wide_k1, wide_stream)
    assert topo.validate_alignment(wide_k1_trunk, wide_stream)

    assert not issubgraph(wide_trunk, tall_stream)
    assert not issubgraph(tall_trunk, wide_stream)


def test_trunk_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)

    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)

    ctrunk = cs.trunk()
    ftrunk = fs.trunk()

    assert isequivalent(ctrunk, ftrunk)


def test_klargestconncomps_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)

    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)

    ck1 = cs.klargestconncomps()
    fk1 = fs.klargestconncomps()

    assert isequivalent(ck1, fk1)


def test_subgraph_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)

    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)

    # Subgraph can take GridObjects/ndarrays as input. Ensure that
    # those work regardless of memory order
    cb = np.array(cdem) > 50
    fb = np.asfortranarray(cb)

    # Try every combination
    csc = cs.subgraph(cb)
    csf = cs.subgraph(fb)
    fsc = fs.subgraph(cb)
    fsf = fs.subgraph(fb)

    assert isequivalent(csc, csf)
    assert isequivalent(csc, fsc)
    assert isequivalent(csc, fsf)


def test_ezgetnal(tall_dem):
    fd = topo.FlowObject(tall_dem)
    s = topo.StreamObject(fd)

    z = s.ezgetnal(tall_dem)
    z2 = s.ezgetnal(z)
    z3 = s.ezgetnal(z, dtype=np.float64)

    # ezgetnal should be idempotent
    assert np.array_equal(z, z2)
    assert np.array_equal(z, z3)

    # ezgetnal should always return a copy
    assert z is not z2
    assert z is not z3

    assert z.dtype == tall_dem.z.dtype
    assert z2.dtype == tall_dem.z.dtype
    # ezgetnal with the dtype argument should return array of that type
    assert z3.dtype is np.dtype(np.float64)


def test_ezgetnal_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)

    cz = cs.ezgetnal(cdem)
    fz = fs.ezgetnal(fdem)

    cdz = np.zeros(cs.shape)
    fdz = np.zeros(fs.shape)

    cdz[cs.node_indices] = cz
    fdz[fs.node_indices] = fz

    assert np.array_equal(cdz, fdz)


def test_streampoi_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)

    ch = cs.streampoi('channelheads')
    co = cs.streampoi('outlets')
    cc = cs.streampoi('confluences')

    fh = fs.streampoi('channelheads')
    fo = fs.streampoi('outlets')
    fc = fs.streampoi('confluences')

    cp = np.zeros(cs.shape, dtype=np.uint32)
    fp = np.zeros(fs.shape, dtype=np.uint32)

    cp[cs.node_indices_where(ch)] |= 1
    cp[cs.node_indices_where(co)] |= 2
    cp[cs.node_indices_where(cc)] |= 4

    fp[fs.node_indices_where(fh)] |= 1
    fp[fs.node_indices_where(fo)] |= 2
    fp[fs.node_indices_where(fc)] |= 4

    assert np.array_equal(cp, fp)


def test_subgraph(tall_dem, wide_dem):
    ############
    # Tall DEM #
    ############

    fd = topo.FlowObject(tall_dem)
    s = topo.StreamObject(fd)

    d = s.downstream_distance()
    nal = d > 10

    sub = s.subgraph(nal)

    # The subgraph should have no more vertices than are true in the
    # node attribute list.
    assert np.size(sub.stream) <= np.count_nonzero(nal)

    # The subgraph should have no more edges than the original network
    assert np.size(sub.source) <= np.size(s.source)

    # The nodes of the subgraph should be a subset of the nodes of the
    # original graph
    assert set(sub.stream) <= set(s.stream)

    # The edges of the subgraph should be a subset of the edges of the
    # original graph. This is what issubgraph tests.
    assert issubgraph(sub, s)

    # Subgraph should be idempotent
    nal_all = np.ones(s.stream.size,dtype=np.bool)
    sub_all = s.subgraph(nal_all)
    assert issubgraph(sub, s)
    assert np.array_equal(sub_all.stream, s.stream)
    assert np.array_equal(sub_all.source, s.source)
    assert np.array_equal(sub_all.target, s.target)

    # Subgraph of an empty subset should be empty
    nal_none = np.zeros(s.stream.size, dtype=np.bool)
    sub_none = s.subgraph(nal_none)
    assert issubgraph(sub, s)
    assert np.size(sub_none.stream) == 0
    assert np.size(sub_none.source) == 0
    assert np.size(sub_none.target) == 0

    ############
    # Wide DEM #
    ############

    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)

    d = s.downstream_distance()
    nal = d > 10

    sub = s.subgraph(nal)

    # The subgraph should have no more vertices than are true in the
    # node attribute list.
    assert np.size(sub.stream) <= np.count_nonzero(nal)

    # The subgraph should have no more edges than the original network
    assert np.size(sub.source) <= np.size(s.source)

    # The nodes of the subgraph should be a subset of the nodes of the
    # original graph
    assert set(sub.stream) <= set(s.stream)

    # The edges of the subgraph should be a subset of the edges of the
    # original graph. This is what issubgraph tests.
    assert issubgraph(sub, s)

    # Subgraph should be idempotent
    nal_all = np.ones(s.stream.size,dtype=np.bool)
    sub_all = s.subgraph(nal_all)
    assert issubgraph(sub, s)
    assert np.array_equal(sub_all.stream, s.stream)
    assert np.array_equal(sub_all.source, s.source)
    assert np.array_equal(sub_all.target, s.target)

    # Subgraph of an empty subset should be empty
    nal_none = np.zeros(s.stream.size, dtype=np.bool)
    sub_none = s.subgraph(nal_none)
    assert issubgraph(sub, s)
    assert np.size(sub_none.stream) == 0
    assert np.size(sub_none.source) == 0
    assert np.size(sub_none.target) == 0

def test_stream_channelheads(tall_dem, wide_dem):
    fd = topo.FlowObject(tall_dem)
    s = topo.StreamObject(fd)

    assert topo.validate_alignment(fd, s)

    channel_heads = s.streampoi("channelheads")

    s2 = topo.StreamObject(fd, channelheads=s.node_indices_where(channel_heads))

    assert topo.validate_alignment(fd, s2)

    assert np.array_equal(s2.stream[s2.source], s.stream[s.source])
    assert np.array_equal(s2.stream[s2.target], s.stream[s.target])

    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)

    assert topo.validate_alignment(fd, s)

    channel_heads = s.streampoi("channelheads")

    s2 = topo.StreamObject(fd, channelheads=s.node_indices_where(channel_heads))

    assert topo.validate_alignment(fd, s2)

    assert np.array_equal(s2.stream[s2.source], s.stream[s.source])
    assert np.array_equal(s2.stream[s2.target], s.stream[s.target])

def test_streamobject_ch_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)

    cch = cs.streampoi("channelheads")
    fch = fs.streampoi("channelheads")

    cs2 = topo.StreamObject(cfd, channelheads=cs.node_indices_where(cch))
    fs2 = topo.StreamObject(ffd, channelheads=fs.node_indices_where(fch))

    assert isequivalent(cs2,fs2)

def test_stream_downstreamto(tall_dem):
    fd = topo.FlowObject(tall_dem)
    s = topo.StreamObject(fd)

    ch = s.streampoi("channelheads")

    sc = topo.StreamObject(fd,channelheads=s.node_indices_where(ch))
    sd = s.downstreamto(ch)

    # These two stream networks should be equivalent
    assert len(set(sd.stream).symmetric_difference(set(sc.stream))) == 0

def test_stream_upstreamto(tall_dem):
    fd = topo.FlowObject(tall_dem)

    # We clean here in case any 1 pixel streams exist in s.
    # They won't be identified as outlets, so the reconstructed stream
    # network would not be identical to the original.
    s = topo.StreamObject(fd).clean()

    outlets = s.streampoi("outlets")

    su = s.upstreamto(outlets)

    # These two stream networks should be equivalent
    assert len(set(s.stream).symmetric_difference(set(su.stream))) == 0

def test_upstreamto_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd).clean()
    fs = topo.StreamObject(ffd).clean()

    b = np.array(cdem) > 50

    cb = cs.upstreamto(b)
    fb = fs.upstreamto(b)

    # These assertions make sure that this test is doing
    # something. Otherwise we may need to change the threshold above.
    assert not isequivalent(cb, cs)
    assert not isequivalent(fb, fs)

    assert isequivalent(cb, fb)

def test_downstreamto_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)

    b = np.array(cdem) > 50

    cb = cs.downstreamto(b)
    fb = fs.downstreamto(b)

    assert not isequivalent(cb, cs)
    assert not isequivalent(fb, fs)

    assert isequivalent(cb, fb)

def test_stream_imposemin(tall_dem, wide_dem):
    fd = topo.FlowObject(tall_dem)
    s = topo.StreamObject(fd)

    original_z = s.ezgetnal(tall_dem)

    for minimum_slope in [0.0,0.001,0.01,0.1]:
        minz = topo.imposemin(s, tall_dem, minimum_slope)

        # imposemin should not modify the original array
        assert np.array_equal(original_z, s.ezgetnal(tall_dem))

        # The carved dem should not be above the original
        assert np.all(minz <= s.ezgetnal(tall_dem))

        # The gradient along the flow network should be greater than or
        # equal to the defined slope within some numerical error
        g = (minz[s.source] - minz[s.target])/s.distance()
        assert np.all(g >= minimum_slope - 1e-6)

    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)

    original_z = s.ezgetnal(wide_dem)

    for minimum_slope in [0.0,0.001,0.01,0.1]:
        minz = topo.imposemin(s, wide_dem, minimum_slope)

        # imposemin should not modify the original array
        assert np.array_equal(original_z, s.ezgetnal(wide_dem))

        # The carved dem should not be above the original
        assert np.all(minz <= s.ezgetnal(wide_dem))

        # The gradient along the flow network should be greater than or
        # equal to the defined slope within some numerical error
        g = (minz[s.source] - minz[s.target])/s.distance()
        assert np.all(g >= minimum_slope - 1e-6)

def test_imposemin_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)


    cminslope = topo.imposemin(cs, cdem, minimum_slope=0.001)
    cz = np.zeros_like(cdem)
    cz[cs.node_indices] = cminslope

    fminslope = topo.imposemin(fs, fdem, minimum_slope=0.001)
    fz = np.zeros_like(fdem)
    fz[fs.node_indices] = fminslope

    assert np.array_equal(cz, fz)

def test_gradient(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)
    g = s.gradient(wide_dem, impose = True)

    assert np.all(g >= 0)

def test_ksn(wide_dem):
    fd = topo.FlowObject(wide_dem)
    A = fd.flow_accumulation()
    s = topo.StreamObject(fd)
    theta = 0.45
    g = s.gradient(wide_dem, impose = True)
    k = s.ksn(wide_dem, A, theta)

def test_streamorder(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)
    cs = topo.StreamObject(cfd)
    fs = topo.StreamObject(ffd)

    cds = cs.streamorder()
    fds = fs.streamorder()

    cdg = np.zeros(cfd.shape)
    fdg = np.zeros(ffd.shape)

    cdg[cs.node_indices] = cds
    fdg[fs.node_indices] = fds

    assert np.array_equal(cdg, fdg)

def test_crslin(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)
    s = s.klargestconncomps()

    z = s.ezgetnal(wide_dem, dtype = 'double')
    zs = s.crslin(wide_dem, k = 1, mingradient = 0.01, attachtomin=True, attachheads=True)

    # attachtomin: zs <= z
    assert np.all(zs <= z)

    # attachheads: zs(channelheads) = z(channelheads)
    channelheads = s.streampoi('channelheads')
    assert np.all(zs[channelheads] == z[channelheads])

    # gradient: gradient > mingradient
    gradient = (zs[s.source] - zs[s.target]) / s.distance()
    mingradient = 0.01
    assert np.all((gradient - mingradient) >= -1e-6)

def test_crslin_noattachheads(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)
    s = s.klargestconncomps()

    z = s.ezgetnal(wide_dem, dtype='double')
    zs = s.crslin(wide_dem, k = 1, mingradient = 0.01, attachtomin=True, attachheads=False)

    assert np.all(zs <= z)

    # gradient: gradient > mingradient
    gradient = (zs[s.source] - zs[s.target]) / s.distance()
    mingradient = 0.01
    assert np.all((gradient - mingradient) >= -1e-6)

def test_quantcarve(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)
    s = s.klargestconncomps()
    z = s.ezgetnal(wide_dem, dtype = 'double')

    zs = s.quantcarve(wide_dem, tau = 0.5, mingradient = 0.01, fixedoutlet = True)

    # outlets: zs[outlets] = z[outlets]
    outlets = s.streampoi('outlets')
    assert np.all(zs[outlets] == zs[outlets])

    # gradient: gradient > mingradient
    gradient = (zs[s.source] - zs[s.target]) / s.distance()
    mingradient = 0.01
    assert np.all((gradient - mingradient) >= -1e-6)

def test_lowerenv_convex(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd)
    s = s.klargestconncomps(1)

    z = topo.imposemin(s, wide_dem)

    kn = np.arange(len(z)) % 13 == 0

    ze = s.lowerenv(z, kn)

    g = np.zeros(len(z))
    g[s.source] = (ze[s.source] - ze[s.target]) / s.distance()

    # The gradient of any incoming edge to a node should be greater
    # than the gradient of its outgoing edge, with some room for
    # numerical errors, except at knickpoints
    assert np.all((g[s.source] >= g[s.target] - 1e-6) + kn[s.target])


def test_lowerenv_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)
    cs = cs.klargestconncomps(1)

    cz = topo.imposemin(cs, cdem)

    ckn = np.arange(len(cz)) % 13 == 0

    cze = cs.lowerenv(cz, ckn)

    cg = np.zeros(len(cz))
    cg[cs.source] = (cze[cs.source] - cze[cs.target]) / cs.distance()

    # The gradient of any incoming edge to a node should be greater
    # than the gradient of its outgoing edge, with some room for
    # numerical errors, except at knickpoints
    assert np.all((cg[cs.source] >= cg[cs.target] - 1e-5) + ckn[cs.target])

    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)
    fs = fs.klargestconncomps(1)

    fz = topo.imposemin(fs, fdem)

    # Make sure the same knickpoints are used in both arrays
    kn = np.zeros(cs.shape, dtype=np.bool)
    kn[cs.node_indices] = ckn
    fkn = kn[fs.node_indices]

    fze = fs.lowerenv(fz, fkn)

    fg = np.zeros(len(fz))
    fg[fs.source] = (fze[fs.source] - fze[fs.target]) / fs.distance()

    assert np.all((fg[fs.source] >= fg[fs.target] - 1e-5) + fkn[fs.target])

    czz = np.zeros(cs.shape)
    fzz = np.zeros(fs.shape)

    czz[cs.node_indices] = cze
    fzz[fs.node_indices] = fze

    assert np.allclose(czz, fzz)

def test_conncomps(wide_dem):
    fd = topo.FlowObject(wide_dem)
    s = topo.StreamObject(fd).clean()
    l = s.conncomps()

    # Each node is in the same connected component as its downstream
    # neighbor.
    assert np.array_equal(l[s.source], l[s.target])

    # There should be at least one connected component
    assert np.max(l) > 0
    # The minimum label should be 1
    assert np.min(l) == 1

def test_conncomps_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    cs = topo.StreamObject(cfd)

    ffd = topo.FlowObject(fdem)
    fs = topo.StreamObject(ffd)

    clg = np.zeros(cfd.shape)
    clg[cs.node_indices] = cs.conncomps()

    flg = np.zeros(ffd.shape)
    flg[fs.node_indices] = fs.conncomps()

    # Labels are identical up to a bijection. See
    # tests/test_flow_object.py (test_drainagebasins_order) for
    # discussion on this approach.
    cu, cind, cinv = np.unique(clg, return_index=True, return_inverse=True)
    fu, find, finv = np.unique(flg, return_index=True, return_inverse=True)

    assert np.array_equal(flg, fu[np.take(finv, cind)][cinv])
    assert np.array_equal(clg, cu[np.take(cinv, find)][finv])
