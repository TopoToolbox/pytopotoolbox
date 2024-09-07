"""This module contains the FlowObject class.
"""
import matplotlib.pyplot as plt
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
        Display the StreamObject instance as an image using Matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap that will be used in the plot.
        """
        plt.imshow(self.z, cmap=cmap)
        plt.title(self.name)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __len__(self):
        return len(self.z)

    def __iter__(self):
        return iter(self.z)

    def __getitem__(self, index):
        return self.z[index]

    def __setitem__(self, index, value):
        try:
            value = np.float32(value)
        except (ValueError, TypeError):
            raise TypeError(
                f"{value} can't be converted to float32.") from None

        self.z[index] = value

    def __array__(self):
        return self.z

    def __str__(self):
        return str(self.z)
