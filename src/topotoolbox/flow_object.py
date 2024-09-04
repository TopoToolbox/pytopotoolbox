"""This module contains the FlowObject class.
"""
import matplotlib.pyplot as plt
import numpy as np

# pylint: disable=import-error
from ._grid import (grid_fillsinks, grid_flow_routing_d8_carve,  # type: ignore
                    grid_flow_routing_targets, grid_gwdt,
                    grid_gwdt_computecosts, grid_identifyflats)

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
        grid_fillsinks(filled_dem, dem, dims)

        flats = np.zeros_like(dem, dtype=np.int32, order='F')
        grid_identifyflats(flats, filled_dem, dims)

        costs = np.zeros_like(dem, dtype=np.float32, order='F')
        conncomps = np.zeros_like(dem, dtype=np.int64, order='F')
        grid_gwdt_computecosts(costs, conncomps, flats, dem, filled_dem, dims)

        dist = np.zeros_like(flats, dtype=np.float32, order='F')
        prev = conncomps  # prev: dtype=np.int64
        heap = np.zeros_like(flats, dtype=np.int64, order='F')
        back = np.zeros_like(flats, dtype=np.int64, order='F')
        grid_gwdt(dist, prev, costs, flats, heap, back, dims)

        source = heap  # source: dtype=np.int64
        direction = np.zeros_like(dem, dtype=np.uint8, order='F')
        grid_flow_routing_d8_carve(
            source, direction, filled_dem, dist, flats, dims)

        target = back  # target: dtype=int64
        grid_flow_routing_targets(target, source, direction, dims)

        # TODO: use 'del' to immediately delete unused large arrays?

        self.path = grid.path
        self.name = grid.name

        # raster metadata

        self.z = filled_dem
        self.target = target
        self.source = source
        self.direction = direction
        self.shape = grid.shape

        # georeference
        self.bounds = grid.bounds
        self.transform = grid.transform
        self.crs = grid.crs

    def show(self, cmap: str = 'terrain'):
        """
        Display the FlowObject instance as an image using Matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap that will be used in the plot.
        """

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(self.z, cmap=cmap)
        axs[0].set_title('dem')

        axs[1].imshow(self.target, cmap=cmap)
        axs[1].set_title('target')

        axs[2].imshow(self.source, cmap=cmap)
        axs[2].set_title('source')

        plt.tight_layout()
        plt.show()

    # TODO: Add magic functions, maybe use mixins to reuse GridObject functions
