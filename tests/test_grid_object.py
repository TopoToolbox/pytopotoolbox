import numpy as np
import pytest

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
    dem_C.z = 64 * (opensimplex.noise2array(x,y) + 1)

    assert dem_C.z.flags.c_contiguous
    
    dem_F = topo.GridObject()
    dem_F.z = np.asfortranarray(dem_C.z)

    assert dem_F.z.flags.f_contiguous

    filled_C = dem_C.fillsinks()
    assert filled_C.z.flags.c_contiguous
    
    filled_F = dem_F.fillsinks()    
    assert filled_F.z.flags.f_contiguous
   
    assert np.array_equal(filled_F.z, filled_C.z)
    
def test_identifyflats(square_dem, wide_dem, tall_dem):
    # TODO: add more tests
    for dem in [square_dem, wide_dem, tall_dem]:
        sills, flats = dem.identifyflats()

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

    def test_excesstopography(square_dem):
        # TODO: add more tests
        assert square_dem.excesstopography(threshold='0.1') == TypeError()
