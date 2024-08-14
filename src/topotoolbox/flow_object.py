"""This module contains the FlowObject class.
"""

import copy

import numpy as np
import matplotlib.pyplot as plt
from .grid_object import *

# pylint: disable=import-error
# from ._flow import (  # type: ignore
#    flow_somefuncname
# )

# pylint: disable=import-error
from ._grid import (  # type: ignore
    grid_fillsinks,
    grid_identifyflats,
    grid_gwdt,
    grid_gwdt_computecosts,
    grid_flow_routing_d8_carve,
    grid_flow_routing_targets
)

__all__ = ['FlowObject']


class FlowObject():
    def __init__(self, grid: GridObject):
        dims = grid.shape
        dem = grid.z

        filled_dem = np.zeros_like(dem, dtype=np.float32, order='F')
        grid_fillsinks(filled_dem, dem, dims)

        flats = np.zeros_like(dem, dtype=np.int32, order='F')
        grid_identifyflats(flats, filled_dem, dims)

        costs = np.zeros_like(dem, dtype=np.float32, order='F')
        conncomps = np.zeros_like(dem, dtype=np.int64, order='F')
        grid_gwdt_computecosts(costs, conncomps, flats, dem, filled_dem, dims)

        dist = np.zeros_like(flats, dtype=np.float32)
        # prev = np.zeros_like(flats, dtype=np.int64)
        prev = conncomps
        heap = np.zeros_like(flats, dtype=np.int64)
        back = np.zeros_like(flats, dtype=np.int64)
        grid_gwdt(dist, prev, costs, flats, heap, back, dims)

        # source = np.zeros_like(dem, dtype=np.int64, order='F')
        source = heap
        direction = np.zeros_like(dem, dtype=np.uint8, order='F')
        grid_flow_routing_d8_carve(
            source, direction, filled_dem, dist, flats, dims)

        # target = np.zeros_like(dem, dtype=np.int64, order='F')
        target = back
        grid_flow_routing_targets(target, source, direction, dims)
        # TODO: use del to delete unused large arrays?

        self.path = grid.path
        self.name = grid.name

        # raster metadata
        self.z = dem
        self.target = target
        self.source = source
        self.shape = grid.shape

        # georeference
        self.bounds = grid.bounds
        self.transform = grid.transform
        self.crs = grid.crs
