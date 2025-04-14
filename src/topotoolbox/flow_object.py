"""This module contains the FlowObject class.
"""
import numpy as np

# pylint: disable=no-name-in-module
from . import _grid  # type: ignore
from . import _flow  # type: ignore
from . import _stream  # type: ignore
from .grid_object import GridObject
from .utils import validate_alignment

__all__ = ['FlowObject']


class FlowObject():
    """A class containing containing (water-) flow information about a given
    digital elevation model (DEM).
    """

    def __init__(self, grid: GridObject,
                 bc: np.ndarray | GridObject | None = None,
                 hybrid: bool = True):
        """The constructor for the FlowObject. Takes a GridObject as input,
        computes flow direction information and saves them as an FlowObject.

        Parameters
        ----------
        grid : GridObject
            The GridObject that will be the basis of the computation.
        bc : ndarray or GridObject, optional
            Boundary conditions for sink filling. `bc` should be an array
            of np.uint8 that matches the shape of the DEM. Values of 1
            indicate pixels that should be fixed to their values in the
            original DEM and values of 0 indicate pixels that should be
            filled.
        hybrid: bool, optional
            Should hybrid reconstruction algorithm be used to fill
            sinks? Defaults to True. Hybrid reconstruction is faster
            but requires additional memory be allocated for a queue.

        Notes
        -----
        Large intermediate arrays are created during the initialization
        process, which could lead to issues when using very large DEMs.
        """
        dims = grid.dims
        dem = np.asarray(grid, dtype=np.float32)

        filled_dem = np.zeros_like(dem, dtype=np.float32)
        restore_nans = False
        if bc is None:
            bc = np.ones_like(dem, dtype=np.uint8)
            bc[1:-1, 1:-1] = 0  # Set interior pixels to 0

            nans = np.isnan(dem)
            dem[nans] = -np.inf
            bc[nans] = 1
            restore_nans = True

        if not validate_alignment(grid, bc):
            err = ("The shape of the provided boundary conditions does not "
                   f"match the shape of the DEM. {dims}")
            raise ValueError(err)from None

        bc = np.asarray(bc, dtype=np.uint8)

        queue = np.zeros_like(dem, dtype=np.int64)
        if hybrid:
            _grid.fillsinks_hybrid(filled_dem, queue, dem, bc, dims)
        else:
            _grid.fillsinks(filled_dem, dem, bc, dims)

        if restore_nans:
            dem[nans] = np.nan
            filled_dem[nans] = np.nan

        flats = np.zeros_like(dem, dtype=np.int32)
        _grid.identifyflats(flats, filled_dem, dims)

        costs = np.zeros_like(dem, dtype=np.float32)
        conncomps = np.zeros_like(dem, dtype=np.int64)
        _grid.gwdt_computecosts(costs, conncomps, flats, dem, filled_dem, dims)

        dist = np.zeros_like(flats, dtype=np.float32)
        prev = conncomps  # prev: dtype=np.int64
        heap = queue      # heap: dtype=np.int64
        back = np.zeros_like(flats, dtype=np.int64)
        _grid.gwdt(dist, prev, costs, flats, heap, back, dims)

        node = heap  # node: dtype=np.int64
        direction = np.zeros_like(dem, dtype=np.uint8)
        _grid.flow_routing_d8_carve(
            node, direction, filled_dem, dist, flats, dims)

        # ravel is used here to flatten the arrays. The memory order should not matter
        # because we only need a block of contiguous memory interpreted as a 1D array.
        source = np.ravel(conncomps)  # source: dtype=int64
        target = np.ravel(back)       # target: dtype=int64
        edge_count = _grid.flow_routing_d8_edgelist(
            source, target, node, direction, dims)

        self.path = grid.path
        self.name = grid.name

        # raster metadata
        self.direction = direction  # dtype=np.unit8

        self.source = source[0:edge_count]  # dtype=np.int64
        self.target = target[0:edge_count]  # dtype=np.int64

        self.shape = grid.shape
        self.cellsize = grid.cellsize
        self.strides = tuple(s // grid.z.itemsize for s in grid.z.strides)

        # georeference
        self.bounds = grid.bounds
        self.transform = grid.transform
        self.crs = grid.crs

    def ezgetnal(self, k, dtype=None):
        """Retrieve a node attribute list

        Parameters
        ----------
        k : GridObject or np.ndarray or scalar        
            The object from which node values will be extracted. If
            `k` is a `GridObject` or an `ndarray` with the same shape
            as this `FlowObject`, then a copy is returned. If it is a
            scalar, an `ndarray` with the appropriate shape, filled
            with `k`, is returned.

        Returns
        -------
        GridObject or np.ndarray

        Raises
        ------
        ValueError
            The supplied input is not aligned with the FlowObject.

        """
        if np.isscalar(k):
            return np.full(self.shape, k, dtype=dtype)
        if not validate_alignment(self, k):
            raise ValueError("Input is not properly aligned to the FlowObject")

        return k.astype(dtype)

    def flow_accumulation(self, weights: np.ndarray | float = 1.0):
        """Computes the flow accumulation for a given flow network using
        optional weights. The flow accumulation represents the amount of flow
        each cell receives from its upstream neighbors.

        Parameters
        ----------
        weights : np.ndarray | float, optional
            An array of the same shape as the flow grid representing weights
            for each cell, or a constant float value used as the weight for all
            cells. If `weights=1.0` (default), the flow accumulation is
            unweighted. If an ndarray is provided, it must match the shape of
            the flow grid., by default 1.0

        Returns
        -------
        GridObject
            A new GridObject containing the flow accumulation grid.

        Raises
        ------
        ValueError
            If the shape of the `weights` array does not match the shape of the
            flow network grid.
        """
        acc = np.zeros(self.shape, dtype=np.float32, order='F')

        if weights == 1.0:
            weights = np.ones(self.shape, dtype=np.float32, order='F')
        elif isinstance(weights, np.ndarray):
            if weights.shape != acc.shape:
                err = ("The shape of the provided weights ndarray does not "
                       f"match the shape of the FlowObject. {self.shape}")
                raise ValueError(err)from None
        else:
            weights = np.full(self.shape, weights, dtype=np.float32, order='F')

        fraction = np.ones_like(self.source, dtype=np.float32)

        _flow.flow_accumulation(
            acc, self.source, self.target, fraction, weights, self.shape)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = acc
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.crs = self.crs

        return result

    def drainagebasins(self):
        """Delineate drainage basins from a flow network.

        Returns
        -------
        GridObject
            An integer-valued GridObject with a unique label for each drainage
            basin.
        """
        basins = np.zeros(self.shape, dtype=np.int64, order='F')

        _flow.drainagebasins(basins, self.source, self.target, self.shape)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = basins
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.crs = self.crs

        return result

    def flowpathextract(self, idx: int):
        """Extract linear indices of a single flowpath in a DEM

        The flow path downstream of idx is extracted from the flow
        directions recorded in FlowObject.

        Parameters
        ----------
        idx: int
            The column-major linear index of the starting pixel of the flowpath.

        Returns
        -------
        np.ndarray        
            An array containing column-major linear indices into the
            DEM identifying the flow path.
        """
        ch = np.zeros(self.shape, dtype=np.uint32, order='F')
        ch[np.unravel_index(idx, self.shape, order='F')] = 1
        edges = np.ones(self.source.size, dtype=np.uint32)
        _stream.traverse_down_u32_or_and(ch, edges, self.source, self.target)

        return np.nonzero(np.ravel(ch, order='F'))[0]

    def distance(self):
        """Compute the distance between each node in the flow network

        Returns
        -------
        np.ndarray
            An array (edge attribute list) with the interpixel
            distance. This will be either cellsize or sqrt(2)*cellsize
        """
        d = np.abs(self.source - self.target)
        dist = self.cellsize * np.where(
            (d == self.strides[0]) | (d == self.strides[1]),
            np.float32(1.0),
            np.sqrt(np.float32(2.0)))
        return dist

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
            if array_name == 'source':
                return self.source[idx]
            if array_name == 'direction':
                return self.direction[idx]
            raise ValueError(
                "Invalid raster name ('target', 'source', or 'direction').")
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
