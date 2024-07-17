import numpy as np
import pytest

import topotoolbox.grid_object as topo


@pytest.fixture
def squareGridObject():
    grid = topo.GridObject()
    grid.z = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    grid.rows, grid.columns = grid.z.shape
    grid.shape = grid.z.shape
    return grid


@pytest.fixture
def tallGridObject():
    grid = topo.GridObject()
    grid.z = np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    grid.rows, grid.columns = grid.z.shape
    grid.shape = grid.z.shape
    return grid


@pytest.fixture
def wideGridObject():
    grid = topo.GridObject()
    grid.z = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1, 0, 1],
    ])
    grid.rows, grid.columns = grid.z.shape
    grid.shape = grid.z.shape
    return grid


def test_fillsinks():
    assert True


def test_identifyflats():
    pass
