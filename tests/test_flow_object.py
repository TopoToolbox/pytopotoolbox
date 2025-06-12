import copy
import pytest

import numpy as np
import opensimplex

import topotoolbox as topo


def isequivalent(s1 : topo.FlowObject, s2 : topo.FlowObject):
    e1 = set(zip(zip(*s1.source_indices),zip(*s1.target_indices)))
    e2 = set(zip(zip(*s2.source_indices),zip(*s2.target_indices)))

    return (s1.shape == s2.shape) and (e1 == e2)


@pytest.fixture
def wide_dem():
    return topo.gen_random(rows=64, columns=128, seed=12)


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


def test_flowobject(wide_dem):
    dem = wide_dem
    original_dem = dem.z.copy()
    fd = topo.FlowObject(dem)

    assert topo.validate_alignment(dem, fd)

    # Ensure that the source and target arrays contain valid pixel indices
    assert np.all((0 <= fd.source) & (fd.source < np.prod(dem.shape)))
    assert np.all((0 <= fd.target) & (fd.target < np.prod(dem.shape)))

    # Run flow_accumulation at least once during the tests
    acc = fd.flow_accumulation()

    assert topo.validate_alignment(fd, acc)

    # Ensure that FlowObject does not modify the original DEM
    assert np.all(dem.z == original_dem)


def test_flowobject_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    assert isequivalent(cfd, ffd)


def test_flow_accumulation_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    ca = cfd.flow_accumulation()
    fa = ffd.flow_accumulation()

    assert np.array_equal(ca, fa)


def test_drainagebasins_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cdb = cfd.drainagebasins()
    fdb = ffd.drainagebasins()

    # This testing logic is rather complicated, so please excuse the
    # length explanation.

    # The two sets of labels are identical up to a bijection between
    # the label sets. We construct that bijection here. We need the
    # set of unique labels, `us`, the indices of a representative pixel
    # in the drainage basins array that has each label, `idxs`, and, for
    # each pixel, the index of its label in the label set, `invs`. If we
    # have B basins in an N x M DEM, `us` is a length B array of basin
    # labels, `idxs` is an length B array of pixel indices and `invs` is
    # an N x M array of the indices [0,B-1) into the `us`
    # array. Fortunately np.unique can provide all of these.
    cus, cidxs, cinvs = np.unique(cdb, return_index=True, return_inverse=True)
    fus, fidxs, finvs = np.unique(fdb, return_index=True, return_inverse=True)

    # Now we construct the bijection and apply it to reconstruct the
    # column-major and row-major drainage basins. Below the "main array"
    # refers to the array whose pattern we are trying to reconstruct
    # and the "alternate array" is the one we reconstruct from.

    # First, we use the `idxs` of the alternate array to extract the
    # `invs` of the main array at each of the representative
    # pixels. Then we index into the `us` of the main array to find
    # the labels assigned to each of the representative pixels /in the
    # order in which they show up in the alternate `idxs` array/,
    # which is also the order of the labels in the alternate `us`
    # array. This array represents the bijection between the main
    # labels and the alternate labels. When we index into this length
    # B array with the alternate `invs`, we will get the main labels
    # that should be assigned to each pixel in the alternate array if
    # the two arrays are identical up to the bijection.
    frec = fus[np.take(finvs, cidxs)][cinvs]

    # Now we test that the reconstructed array is identical to the
    # main array. We use flatten because older versions of numpy
    # return different `invs` of different dimensions.
    assert np.array_equal(frec.flatten(), fdb.z.flatten())

    # Having done this once, we might as well do it with the main and
    # alternate arrays swapped.
    crec = cus[np.take(cinvs, fidxs)][finvs]
    assert np.array_equal(crec.flatten(), cdb.z.flatten())


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

    for c in s.stream[ch]:

        s2 = topo.StreamObject(fd, channelheads=fd.unravel_index(c))

        assert topo.validate_alignment(wide_dem, s2)

        idxs = fd.flowpathextract(c)

        # NOTE(wkearn): StreamObject's stream nodes are not
        # necessarily topologically sorted, whereas the flow path
        # returned by flowpathextract is topologically sorted. The two
        # arrays will contain the same elements but in different
        # orders. We therefore verify that idxs is the one and only
        # flow path in s2 by reconstructing the topological order of
        # s2.stream from the edge list.
        assert len(idxs) == len(s2.source) + 1
        for e in np.arange(len(s2.source)):
            u = s2.stream[s2.source[e]]
            v = s2.stream[s2.target[e]]
            assert idxs[e] == u
            assert idxs[e + 1] == v

def test_flowpathextract_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    # Convert the row-major linear index to column-major
    ci = np.ravel_multi_index((37, 109), cfd.shape, order=cfd.order)
    fi = np.ravel_multi_index((37, 109), ffd.shape, order=ffd.order)

    cp = cfd.flowpathextract(ci)
    fp = ffd.flowpathextract(fi)

    # Convert the column-major flow path indices to row-major
    fpc = np.ravel_multi_index(ffd.unravel_index(fp),
                               cfd.shape, order=cfd.order)

    assert np.array_equal(cp, fpc)

def test_distance_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cd = cfd.distance()
    fd = ffd.distance()

    cdg = np.zeros(cfd.shape)
    fdg = np.zeros(ffd.shape)

    cdg[cfd.source_indices] = cd
    fdg[ffd.source_indices] = fd

    assert np.array_equal(cdg, fdg)

def test_imposemin(wide_dem):
    original_dem = wide_dem.z.copy()
    fd = topo.FlowObject(wide_dem)

    g0 = (wide_dem.z[fd.source_indices] -
          wide_dem.z[fd.target_indices])/fd.distance()

    # Make sure that the test array has slopes less than the imposed minimum
    assert not np.all(g0 >= 0.1 - 1e-6)

    for minimum_slope in [0.0, 0.001, 0.01, 0.1]:
        min_dem = topo.imposemin(fd, wide_dem, minimum_slope)

        # The carved dem should not be above the original
        assert np.all(min_dem.z <= wide_dem)

        # The gradient along the flow network should be greater than or
        # equal to the defined slope within some numerical error
        g = (min_dem.z[fd.source_indices] -
             min_dem.z[fd.target_indices])/fd.distance()
        assert np.all(g >= minimum_slope - 1e-6)

        # imposemin should not modify the original array
        assert np.array_equal(original_dem, wide_dem.z)


def test_imposemin_f64(wide_dem):
    original_dem = np.array(wide_dem, dtype=np.float64)

    fd = topo.FlowObject(wide_dem)

    z = np.array(wide_dem.z, dtype=np.float64)

    min_dem = topo.imposemin(fd, z, 0.001)

    assert np.all(min_dem <= z)

    g = (min_dem[fd.source_indices] -
         min_dem[fd.target_indices])/fd.distance()
    assert np.all(g >= 0.001 - 1e-6)

    # imposemin should not modify the original array
    assert np.array_equal(original_dem, z)

def test_imposemin_order(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cminslope = topo.imposemin(cfd, cdem, minimum_slope=0.001)
    fminslope = topo.imposemin(ffd, fdem, minimum_slope=0.001)

    assert np.array_equal(cminslope, fminslope)

def test_downstream_distance(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cd = cfd.downstream_distance()
    fd = ffd.downstream_distance()

    assert np.array_equal(cd, fd)

def test_upstream_distance(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cd = cfd.upstream_distance()
    fd = ffd.upstream_distance()

    assert np.array_equal(cd, fd)

def test_dependence_map(order_dems):
    cdem, fdem = order_dems
    l_c = cdem.duplicate_with_new_data(np.zeros(cdem.shape, dtype = bool, order = 'C'))
    l_f = fdem.duplicate_with_new_data(np.zeros(fdem.shape, dtype = bool, order = 'F'))
    l_c.z[50:100,50:100] = True
    l_f.z[50:100,50:100] = True

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cd = cfd.dependencemap(l_c)
    fd = ffd.dependencemap(l_f)

    assert np.array_equal(cd, fd)

def test_influence_map(order_dems):
    cdem, fdem = order_dems
    l_c = cdem.duplicate_with_new_data(np.zeros(cdem.shape, dtype = bool, order = 'C'))
    l_f = fdem.duplicate_with_new_data(np.zeros(fdem.shape, dtype = bool, order = 'F'))
    l_c.z[50:100,50:100] = True
    l_f.z[50:100,50:100] = True

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cd = cfd.influencemap(l_c)
    fd = ffd.influencemap(l_f)

    assert np.array_equal(cd, fd)
