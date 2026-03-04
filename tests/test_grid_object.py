import numpy as np
import pytest
from rasterio import Affine, CRS
from rasterio.coords import BoundingBox

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
    top_left = cdem.transform * (x[0], y[0])
    bottom_right = cdem.transform * (x[-1], y[-1])
    cdem.bounds = BoundingBox(top=top_left[1], left = top_left[0],
                              bottom=bottom_right[1], right = bottom_right[0])

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0
    fdem.transform = Affine.permutation() * Affine.rotation(90) * Affine.scale(fdem.cellsize)
    fdem.georef = CRS.from_epsg(3857)

    top_left = fdem.transform * (x[0], y[0])
    bottom_right = fdem.transform * (x[-1], y[-1])
    fdem.bounds = BoundingBox(top=top_left[1], left = top_left[0],
                              bottom=bottom_right[1], right = bottom_right[0])

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

@pytest.mark.parametrize("hybrid",[True, False])
@pytest.mark.parametrize("bc",[True, False])
def test_fillsinks_order(order_dems, hybrid, bc):
    cdem, fdem = order_dems

    if bc:
        bc = np.ones_like(cdem.z, dtype=np.uint8)
        bc[1:-1, 1:-1] = 0

        bc = cdem.duplicate_with_new_data(bc)
    else:
        bc = None

    cfilled = cdem.fillsinks(bc=bc, hybrid=hybrid)
    assert cfilled.z.flags.c_contiguous

    ffilled = fdem.fillsinks(bc=bc, hybrid=hybrid)
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


@pytest.fixture(name="threshold_slopes", params=["scalar", "array", "gridobject"])
def threshold_slopes(request, order_dems):
    dem = order_dems[0]

    rng = np.random.default_rng(217412861091418638741329610000239956692)
    t = rng.random(dem.shape)
    if request.param == "scalar":
        return 0.2
    elif request.param == "array":
        return t
    else:
        return dem.duplicate_with_new_data(t)


@pytest.mark.parametrize("method", ['fsm2d', 'fmm2d'])
def test_excesstopography_order(order_dems, method, threshold_slopes):
    cdem, fdem = order_dems

    cext = cdem.excesstopography(threshold=threshold_slopes, method=method)
    assert cext.z.flags.c_contiguous

    fext = fdem.excesstopography(threshold=threshold_slopes, method=method)
    assert fext.z.flags.f_contiguous

    assert np.allclose(fext, cext)

    assert topo.validate_alignment(fdem, fext)
    assert topo.validate_alignment(cdem, cext)


@pytest.mark.parametrize("fused", [True, False])
def test_hillshade_order(order_dems, fused):
    cdem, fdem = order_dems

    for azimuth in np.arange(0.0, 360.0, 2.3):
        hc = cdem.hillshade(fused=fused, azimuth=azimuth)
        hf = fdem.hillshade(fused=fused, azimuth=azimuth)
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

@pytest.mark.parametrize("unit", ['tangent', 'radian', 'degree', 'sine', 'percent'])
@pytest.mark.parametrize("mp", [True, False])
def test_gradient8_order(order_dems, unit, mp):
    cdem, fdem = order_dems

    cgradient = cdem.gradient8(unit=unit, multiprocessing=mp)
    fgradient = fdem.gradient8(unit=unit, multiprocessing=mp)

    assert np.array_equal(cgradient, fgradient)

    assert cgradient.z.flags.c_contiguous
    assert fgradient.z.flags.f_contiguous


@pytest.mark.parametrize("ctype", ['profc', 'planc', 'tangc', 'meanc', 'total'])
@pytest.mark.parametrize("meanfilt", [True, False])
def test_curvature_order(order_dems, ctype, meanfilt):
    cdem, fdem = order_dems

    ccurv = cdem.curvature(meanfilt=meanfilt, ctype=ctype)
    fcurv = fdem.curvature(meanfilt=meanfilt, ctype=ctype)

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


@pytest.mark.parametrize("modified", [True, False])
def test_evansslope_order(order_dems, modified):
    cdem, fdem = order_dems

    cslope = cdem.evansslope(modified=modified)
    fslope = fdem.evansslope(modified=modified)

    assert np.array_equal(cslope, fslope)

    cx, cy = cdem.evansslope(modified=modified, partial_derivatives=True)
    fx, fy = fdem.evansslope(modified=modified, partial_derivatives=True)

    assert np.array_equal(cx, fx)
    assert np.array_equal(cy, fy)

def test_aspect_order(order_dems):
    cdem, fdem = order_dems

    caspect = cdem.aspect()
    faspect = fdem.aspect()

    assert np.array_equal(caspect, faspect)

    caspect_edges = cdem.aspect(classify=True)
    faspect_edges = fdem.aspect(classify=True)

    assert np.array_equal(caspect_edges, faspect_edges)

@pytest.mark.parametrize("hybrid", [True, False])
def test_prominence_order(order_dems, hybrid):
    cdem, fdem = order_dems

    cp, (cx, cy) = cdem.prominence(10.0, use_hybrid=hybrid)
    fp, (fx, fy) = fdem.prominence(10.0, use_hybrid=hybrid)

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

def test_resample_order(order_dems):
    cdem, fdem = order_dems

    c2 = cdem.resample(15.0)
    f2 = fdem.resample(15.0)

    assert np.allclose(f2, c2)

def test_zscore_order(order_dems):
    cdem, fdem = order_dems

    cz = cdem.zscore()
    fz = fdem.zscore()

    assert np.array_equal(cz, fz)

def test_crop_order(order_dems):
    cdem, fdem = order_dems

    c1 = cdem.crop(0.6, 0.8, 0.3, 0.5, 'percent')
    f1 = fdem.crop(0.6, 0.8, 0.3, 0.5, 'percent')

    assert np.array_equal(c1, f1)

    c2 = cdem.crop(20.0, 100.0, -20.0, -60.0, 'coordinate')
    f2 = fdem.crop(20.0, 100.0, -20.0, -60.0, 'coordinate')

    assert np.array_equal(c2, f2)

    c3 = cdem.crop(20, 50, 30, 40, 'pixel')
    f3 = fdem.crop(20, 50, 30, 40, 'pixel')

    assert np.array_equal(c3, f3)
