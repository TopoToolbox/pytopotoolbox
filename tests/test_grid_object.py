import numpy as np
import pytest
from rasterio import Affine, CRS

import topotoolbox as topo

import opensimplex


@pytest.fixture
def square_dem():
    return topo.gen_random(rows=128, columns=128, seed=12)


@pytest.fixture
def wide_dem():
    return topo.gen_random(rows=64, columns=128, seed=12)


@pytest.fixture
def tall_dem():
    return topo.gen_random(rows=128, columns=64, seed=12)


@pytest.fixture
def order_dems():
    opensimplex.seed(12)

    x = np.arange(0, 128)
    y = np.arange(0, 256)

    cdem = topo.GridObject()
    cdem.z = np.array(
        64 * (opensimplex.noise2array(x/13, y/13) + 1), dtype=np.float32)
    cdem.cellsize = 13.0
    cdem.transform = Affine.permutation() * Affine.rotation(90) * Affine.scale(cdem.cellsize)
    cdem.georef = CRS.from_epsg(3857)

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0
    fdem.transform = Affine.permutation() * Affine.rotation(90) * Affine.scale(fdem.cellsize)
    fdem.georef = CRS.from_epsg(3857)

    return [cdem, fdem]


@pytest.fixture
def types_dems():
    opensimplex.seed(12)

    x = np.arange(0, 128)
    y = np.arange(0, 256)

    dem64 = topo.GridObject()
    dem64.z = np.array(
        64 * (opensimplex.noise2array(x/13, y/13) + 1), dtype=np.float64)
    dem64.cellsize = 13.0
    dem64.transform = Affine.scale(dem64.cellsize)

    dem32 = topo.GridObject()
    dem32.z = np.array(dem64, dtype=np.float32)
    dem32.cellsize = 13.0
    dem32.transform = Affine.scale(dem32.cellsize)

    return [dem32, dem64]


def test_fillsinks(square_dem, wide_dem, tall_dem):
    # TODO: add more tests
    for grid in [square_dem, wide_dem, tall_dem]:
        # since grid is a fixture, it has to be assigned/called first
        dem = grid
        original_dem = dem.z.copy()
        filled_dem = dem.fillsinks(hybrid=False)
        filled_dem_hybrid = dem.fillsinks(hybrid=True)

        assert topo.validate_alignment(dem, filled_dem)
        assert topo.validate_alignment(dem, filled_dem_hybrid)

        # Ensure that DEM has not been modified by fillsinks
        assert np.all(original_dem == dem.z)

        # The sequential and hybrid reconstruction algorithms should
        # produce the same result
        assert np.all(filled_dem.z == filled_dem_hybrid.z)

        # Loop over all cells of the DEM
        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):

                # Test: no filled cell lower than before calling fillsinks
                assert dem[i, j] <= filled_dem[i, j]

                # Test: cell isn't a sink
                sink = 0
                for i_offset, j_offset in [
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1)]:

                    i_neighbor = i + i_offset
                    j_neighbor = j + j_offset

                    if (i_neighbor < 0 or i_neighbor >= dem.z.shape[0]
                            or j_neighbor < 0 or j_neighbor >= dem.z.shape[1]):
                        continue

                    if filled_dem[i_neighbor, j_neighbor] > filled_dem[i, j]:
                        sink += 1

                assert sink < 8


def test_fillsinks_order(order_dems):
    cdem, fdem = order_dems

    cfilled = cdem.fillsinks()
    assert cfilled.z.flags.c_contiguous

    ffilled = fdem.fillsinks()
    assert ffilled.z.flags.f_contiguous

    assert np.array_equal(ffilled, cfilled)

    assert topo.validate_alignment(fdem, ffilled)
    assert topo.validate_alignment(cdem, cfilled)


def test_identifyflats(square_dem, wide_dem, tall_dem):
    # TODO: add more tests
    for dem in [square_dem, wide_dem, tall_dem]:
        sills, flats = dem.identifyflats()

        assert topo.validate_alignment(dem, sills)
        assert topo.validate_alignment(dem, flats)

        for i in range(dem.shape[0]):
            for j in range(dem.shape[1]):

                for i_offset, j_offset in [
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1)]:

                    i_neighbor = i + i_offset
                    j_neighbor = j + j_offset

                    if (i_neighbor < 0 or i_neighbor >= dem.z.shape[0]
                            or j_neighbor < 0 or j_neighbor >= dem.z.shape[1]):
                        continue

                    if flats[i_neighbor, j_neighbor] < flats[i, j]:
                        assert flats[i, j] == 1.0


def test_identifyflats_order(order_dems):
    cdem, fdem = order_dems

    cdem_filled = cdem.fillsinks()
    fdem_filled = fdem.fillsinks()

    craw = cdem_filled.identifyflats(raw=True)[0]
    fraw = fdem_filled.identifyflats(raw=True)[0]

    assert np.array_equal(craw, fraw)

    assert craw.flags.c_contiguous
    assert fraw.flags.f_contiguous

    cflats, csills = cdem_filled.identifyflats()
    fflats, fsills = fdem_filled.identifyflats()

    assert np.array_equal(cflats, fflats)
    assert np.array_equal(csills, fsills)

    assert cflats.z.flags.c_contiguous
    assert csills.z.flags.c_contiguous

    assert fflats.z.flags.f_contiguous
    assert fsills.z.flags.f_contiguous


def test_excesstopography(square_dem):
    # TODO: add more tests
    with pytest.raises(TypeError):
        square_dem.excesstopography(threshold='0.1')


def test_excesstopography_order(order_dems):
    cdem, fdem = order_dems

    for method in ['fsm2d', 'fmm2d']:
        cext = cdem.excesstopography(threshold=0.2, method=method)
        assert cext.z.flags.c_contiguous

        fext = fdem.excesstopography(threshold=0.2, method=method)
        assert fext.z.flags.f_contiguous

        assert np.allclose(fext, cext)

        assert topo.validate_alignment(fdem, fext)
        assert topo.validate_alignment(cdem, cext)


def test_hillshade_order(order_dems):
    cdem, fdem = order_dems

    for azimuth in np.arange(0.0, 360.0, 2.3):
        hc = cdem.hillshade(azimuth=azimuth)
        hf = fdem.hillshade(azimuth=azimuth)
        assert np.allclose(hc, hf)

        assert topo.validate_alignment(cdem, hc)
        assert topo.validate_alignment(fdem, hf)


def test_hillshade_fused(wide_dem):
    for azimuth in np.arange(0.0, 360.0, 2.3):
        h1 = wide_dem.hillshade(azimuth=azimuth, fused=True)
        h2 = wide_dem.hillshade(azimuth=azimuth, fused=False)
        assert np.allclose(h1, h2)

        assert topo.validate_alignment(wide_dem, h1)
        assert topo.validate_alignment(wide_dem, h2)


def test_hillshade_types(types_dems):
    dem32, dem64 = types_dems

    for azimuth in np.arange(0.0, 360.0, 2.3):
        h32 = dem32.hillshade(azimuth=azimuth)
        h64 = dem64.hillshade(azimuth=azimuth)
        assert np.allclose(h64, h32)


def test_filter_order(order_dems):
    cdem, fdem = order_dems

    for method in ['mean', 'average', 'median',
                   'sobel', 'scharr', 'wiener', 'std']:
        cfiltered = cdem.filter(method=method)
        ffiltered = fdem.filter(method=method)

        assert np.array_equal(cfiltered, ffiltered)


def test_gradient8_order(order_dems):
    cdem, fdem = order_dems

    cgradient = cdem.gradient8(multiprocessing=True)
    fgradient = fdem.gradient8(multiprocessing=True)

    assert np.array_equal(cgradient, fgradient)

    assert cgradient.z.flags.c_contiguous
    assert fgradient.z.flags.f_contiguous

    cgradient = cdem.gradient8(multiprocessing=False)
    fgradient = fdem.gradient8(multiprocessing=False)

    assert np.array_equal(cgradient, fgradient)

    assert cgradient.z.flags.c_contiguous
    assert fgradient.z.flags.f_contiguous


def test_curvature_order(order_dems):
    cdem, fdem = order_dems

    for ctype in ['profc', 'planc', 'tangc', 'meanc', 'total']:
        ccurv = cdem.curvature(ctype=ctype)
        fcurv = fdem.curvature(ctype=ctype)

        assert np.array_equal(ccurv, fcurv)


def test_dilate_order(order_dems):
    cdem, fdem = order_dems

    cdilated = cdem.dilate((3, 3))
    fdilated = fdem.dilate((3, 3))

    assert np.array_equal(cdilated, fdilated)


def test_erode_order(order_dems):
    cdem, fdem = order_dems

    ceroded = cdem.erode((3, 3))
    feroded = fdem.erode((3, 3))

    assert np.array_equal(ceroded, feroded)


def test_evansslope_order(order_dems):
    cdem, fdem = order_dems

    cslope = cdem.evansslope()
    fslope = fdem.evansslope()

    assert np.array_equal(cslope, fslope)


def test_aspect_order(order_dems):
    cdem, fdem = order_dems

    caspect = cdem.aspect()
    faspect = fdem.aspect()

    assert np.array_equal(caspect, faspect)


def test_prominence(order_dems):
    cdem, fdem = order_dems

    cp, (cx, cy) = cdem.prominence(10.0, use_hybrid=True)
    fp, (fx, fy) = fdem.prominence(10.0, use_hybrid=True)

    assert np.array_equal(cp, fp)
    assert np.array_equal(cx, fx)
    assert np.array_equal(cy, fy)

    cp, (cx, cy) = cdem.prominence(10.0, use_hybrid=False)
    fp, (fx, fy) = fdem.prominence(10.0, use_hybrid=False)

    assert np.array_equal(cp, fp)
    assert np.array_equal(cx, fx)
    assert np.array_equal(cy, fy)


def test_shufflelabel(order_dems):
    cdem, fdem = order_dems

    cfd = topo.FlowObject(cdem)
    ffd = topo.FlowObject(fdem)

    cdb = cfd.drainagebasins()
    cs = cdb.shufflelabel()
    fdb = ffd.drainagebasins()
    fs = fdb.shufflelabel()

    assert cs.shape == cdb.shape
    assert fs.shape == fdb.shape

def test_reproject_order(order_dems):
    cdem, fdem = order_dems

    c2 = cdem.reproject(CRS.from_epsg(32631))
    assert c2.z.flags.c_contiguous

    f2 = fdem.reproject(CRS.from_epsg(32631))
    assert f2.z.flags.f_contiguous

    assert np.allclose(f2, c2, equal_nan=True)
