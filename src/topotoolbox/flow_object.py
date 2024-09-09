"""This module contains the FlowObject class.
"""
import numpy as np

# pylint: disable=no-name-in-module
from . import _grid  # type: ignore
from .grid_object import GridObject

__all__ = ['FlowObject']


class FlowObject():
    """A class containing containing (water-) flow information about a given
    digital elevation model (DEM).
    """

    def __init__(self, grid: GridObject):
        """The constructor for the FlowObject. Takes a GridObject as input,
        computes flow direction information and saves them as an FlowObject.

        Parameters
        ----------
        grid : GridObject
            The GridObject that will be the basis of the computation.

        Notes
        -----
        Large intermediate arrays are created during the initialization
        process, which could lead to issues when using very large DEMs.
        """
        dims = grid.shape
        dem = grid.z

        filled_dem = np.zeros_like(dem, dtype=np.float32, order='F')
        _grid.fillsinks(filled_dem, dem, dims)

        flats = np.zeros_like(dem, dtype=np.int32, order='F')
        _grid.identifyflats(flats, filled_dem, dims)

        costs = np.zeros_like(dem, dtype=np.float32, order='F')
        conncomps = np.zeros_like(dem, dtype=np.int64, order='F')
        _grid.gwdt_computecosts(costs, conncomps, flats, dem, filled_dem, dims)

        dist = np.zeros_like(flats, dtype=np.float32, order='F')
        prev = conncomps  # prev: dtype=np.int64
        heap = np.zeros_like(flats, dtype=np.int64, order='F')
        back = np.zeros_like(flats, dtype=np.int64, order='F')
        _grid.gwdt(dist, prev, costs, flats, heap, back, dims)

        source = heap  # source: dtype=np.int64
        direction = np.zeros_like(dem, dtype=np.uint8, order='F')
        _grid.flow_routing_d8_carve(
            source, direction, filled_dem, dist, flats, dims)

        target = back  # target: dtype=int64
        _grid.flow_routing_targets(target, source, direction, dims)

        self.path = grid.path
        self.name = grid.name

        # raster metadata

        self.target = target  # dtype=np.int64
        self.source = source  # dtype=np.int64
        self.direction = direction  # dtype=np.unit8
        self.shape = grid.shape

        # georeference
        self.bounds = grid.bounds
        self.transform = grid.transform
        self.crs = grid.crs

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __len__(self):
        return len(self.target)

    def __iter__(self):
        return iter(self.target)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            array_name, idx = index

            if array_name == 'target':
                return self.target[idx]
            elif array_name == 'source':
                return self.source[idx]
            elif array_name == 'direction':
                return self.direction[idx]
            else:
                raise ValueError(
                    "Invalid raster name.('target', 'source', or 'direction')")
        else:
            raise ValueError(
                "Index must be a tuple with (raster_name, index).")

    def __setitem__(self, index, value):
        # Check if the index is a tuple
        if isinstance(index, tuple):
            array_name, idx = index

            if array_name == 'target':
                self.target[idx] = value
            elif array_name == 'source':
                self.source[idx] = value
            elif array_name == 'direction':
                self.direction[idx] = value
            else:
                raise ValueError(
                    "Invalid raster name.('target', 'source', or 'direction')")
        else:
            raise ValueError(
                "Index must be a tuple with (raster_name, index).")

    def __array__(self):
        # Concatenate the arrays along their first axis.
        # Not practical to compute with, but if there is a need to manually
        # plot a FlowObject it'll show the logic nicely.
        return np.concatenate((self.target, self.source, self.direction))
