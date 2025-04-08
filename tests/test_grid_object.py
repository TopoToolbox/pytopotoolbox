import numpy as np
import pytest
from rasterio import Affine

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


def test_fillsinks_order():
    opensimplex.seed(12)

    x = np.arange(0,128)
    y = np.arange(0,256)

    dem_C = topo.GridObject()
    dem_C.z = np.array(64 * (opensimplex.noise2array(x,y) + 1), dtype=np.float32)

    assert dem_C.shape[0] == 256
    assert dem_C.shape[1] == 128
    
    assert dem_C.z.flags.c_contiguous
    assert dem_C.dims[0] == 128
    assert dem_C.dims[1] == 256
    
    dem_F = topo.GridObject()
    dem_F.z = np.asfortranarray(dem_C.z)

    assert dem_F.shape[0] == 256
    assert dem_F.shape[1] == 128
    
    assert dem_F.z.flags.f_contiguous
    assert dem_F.dims[0] == 256
    assert dem_F.dims[1] == 128

    filled_C = dem_C.fillsinks()
    assert filled_C.z.flags.c_contiguous
    
    filled_F = dem_F.fillsinks()    
    assert filled_F.z.flags.f_contiguous
   
    assert np.array_equal(filled_F.z, filled_C.z)

    assert topo.validate_alignment(dem_F, filled_F)
    assert topo.validate_alignment(dem_C, filled_C)
    
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

def test_identifyflats_order():
    opensimplex.seed(12)

    x = np.arange(0, 128)
    y = np.arange(0, 256)

    cdem = topo.GridObject()
    cdem.z = np.array(64 * (opensimplex.noise2array(x,y) + 1), dtype=np.float32)
    cdem.cellsize = 13.0
    cdem_filled = cdem.fillsinks()

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0
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

def test_excesstopography_order():
    opensimplex.seed(12)

    x = np.arange(0,128)
    y = np.arange(0,256)

    dem_C = topo.GridObject()
    dem_C.z = np.array(64 * (opensimplex.noise2array(x,y) + 1), dtype=np.float32)
    dem_C.cellsize = 13.0

    assert dem_C.shape[0] == 256
    assert dem_C.shape[1] == 128

    assert dem_C.z.flags.c_contiguous
    assert dem_C.dims[0] == 128
    assert dem_C.dims[1] == 256

    dem_F = topo.GridObject()
    dem_F.z = np.asfortranarray(dem_C.z)
    dem_F.cellsize = 13.0

    assert dem_F.shape[0] == 256
    assert dem_F.shape[1] == 128

    assert dem_F.z.flags.f_contiguous
    assert dem_F.dims[0] == 256
    assert dem_F.dims[1] == 128

    # Compare memory orders using the fast sweeping method
    ext_C = dem_C.excesstopography(threshold=0.2, method='fsm2d')
    assert ext_C.z.flags.c_contiguous

    ext_F = dem_F.excesstopography(threshold=0.2, method='fsm2d')
    assert ext_F.z.flags.f_contiguous

    assert np.array_equal(ext_F.z, ext_C.z)

    # Compare memory orders using the fast marching method
    ext_C = dem_C.excesstopography(threshold=0.2, method='fmm2d')
    assert ext_C.z.flags.c_contiguous

    ext_F = dem_F.excesstopography(threshold=0.2, method='fmm2d')
    assert ext_F.z.flags.f_contiguous

    assert np.array_equal(ext_F.z, ext_C.z)

    assert topo.validate_alignment(dem_F, ext_F)
    assert topo.validate_alignment(dem_C, ext_C)

def test_hillshade_order():
    # The hillshade computed from a column-major array should be
    # identical to that from a row-major array with the same data.

    opensimplex.seed(12)

    x = np.arange(0,128)
    y = np.arange(0,256)

    demc = topo.GridObject()
    demc.z = np.array(64 * (opensimplex.noise2array(x/13, y/13) + 1), dtype=np.float32)
    demc.cellsize = 13.0
    demc.transform = Affine.scale(demc.cellsize)

    # The column-major array gets the same geotransform (following
    # current practice in pytopotoolbox), but is in a different memory
    # order.
    demf = topo.GridObject()
    demf.z = np.asfortranarray(demc.z)
    demf.cellsize = 13.0
    # We also need to permute the geotransform to account for the swapped dimensions
    demf.transform = Affine.rotation(180) * Affine.scale(demf.cellsize)
    
    for azimuth in np.arange(0.0,360.0,2.3):
        hc = demc.hillshade(azimuth=azimuth)
        hf = demf.hillshade(azimuth=azimuth)
        assert np.allclose(hc, hf)

        assert topo.validate_alignment(demc, hc)
        assert topo.validate_alignment(demf, hf)

def test_filter_order():
    opensimplex.seed(12)

    x = np.arange(0,128)
    y = np.arange(0,256)

    cdem = topo.GridObject()
    cdem.z = np.array(64 * (opensimplex.noise2array(x,y) + 1), dtype=np.float32)
    cdem.cellsize = 13.0

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0

    for method in ['mean','average','median',
                   'sobel','scharr','wiener','std']:
        cfiltered = cdem.filter(method=method)
        ffiltered = fdem.filter(method=method)

        assert np.array_equal(cfiltered, ffiltered)

def test_gradient8_order():
    opensimplex.seed(12)

    x = np.arange(0,128)
    y = np.arange(0,256)

    cdem = topo.GridObject()
    cdem.z = np.array(64 * (opensimplex.noise2array(x,y) + 1), dtype=np.float32)
    cdem.cellsize = 13.0

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0

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

def test_curvature_order():
    opensimplex.seed(12)

    x = np.arange(0,128)
    y = np.arange(0,256)

    cdem = topo.GridObject()
    cdem.z = np.array(64 * (opensimplex.noise2array(x,y) + 1), dtype=np.float32)
    cdem.cellsize = 13.0

    fdem = topo.GridObject()
    fdem.z = np.asfortranarray(cdem.z)
    fdem.cellsize = 13.0

    for ctype in ['profc','planc','tangc','meanc','total']:
        ccurv = cdem.curvature(ctype=ctype)
        fcurv = fdem.curvature(ctype=ctype)

        assert np.array_equal(ccurv, fcurv)
