"""This module contains the FlowObject class.
"""
from typing import Literal

import numpy as np

# pylint: disable=no-name-in-module
from . import _grid  # type: ignore
from . import _flow  # type: ignore
from . import _stream  # type: ignore
from .grid_object import GridObject
from .interface import validate_alignment

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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> flow = topotoolbox.FlowObject(dem)
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

        queue = np.zeros(np.prod(dem.shape), dtype=np.int64)
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

        self.stream = node
        self.source = source[0:edge_count]  # dtype=np.int64
        self.target = target[0:edge_count]  # dtype=np.int64

        self.shape = grid.shape
        self.cellsize = grid.cellsize
        self.strides = tuple(s // grid.z.itemsize for s in grid.z.strides)
        self.order: Literal['F', 'C'] = ('F' if grid.z.flags.f_contiguous
                                         else 'C')

        # georeference
        self.bounds = grid.bounds
        self.transform = grid.transform
        self.georef = grid.georef

    @property
    def dims(self):
        """The dimensions of the grid in the correct order for libtopotoolbox
        """
        if self.order == 'C':
            return (self.shape[0], self.shape[1])

        return (self.shape[1], self.shape[0])

    @property
    def source_indices(self) -> tuple[np.ndarray, ...]:
        """The row and column indices of the sources of each edge in
        the flow network.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices of the sources of each edge in the flow
        network. Each of these arrays is an edge attribute lists and
        have a length equal to the number of edges in the flow
        network. This tuple of arrays is suitable for indexing
        GridObjects or arrays shaped like the GridObject from which
        this FlowObject was derived.

        """
        return np.unravel_index(self.source, self.shape, self.order)

    @property
    def target_indices(self) -> tuple[np.ndarray, ...]:
        """The row and column indices of the targets of each edge in
        the flow network.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices of the sources of each edge in the flow
        network. Each of these arrays is an edge attribute lists and
        have a length equal to the number of edges in the flow
        network. This tuple of arrays is suitable for indexing
        GridObjects or arrays shaped like the GridObject from which
        this FlowObject was derived.

        """
        return np.unravel_index(self.target, self.shape, self.order)

    def unravel_index(self, idxs: int | np.ndarray) -> tuple[np.ndarray, ...]:
        """Unravel the provided linear indices so they can be used to
        index grids.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices of the sources of each pixel in the idxs array.
        """
        return np.unravel_index(idxs, self.shape, self.order)

    def ezgetnal(self, k, dtype=None) -> GridObject | np.ndarray:
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

        Example
        -------
        >>> dem = topotoolbox.load_dem('bigtujunga)
        >>> fd = tt3.FlowObject(dem)
        >>> fd.ezgetnal(dem).plot()
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

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> acc = fd.flow_accumulation()
        >>> acc.plot(cmap='Blues',norm="log")
        """
        acc = np.zeros(self.shape, dtype=np.float32, order=self.order)

        w = self.ezgetnal(weights, dtype=np.float32)

        fraction = np.ones_like(self.source, dtype=np.float32)

        _flow.flow_accumulation(
            acc, self.source, self.target, fraction, w, self.shape)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = acc
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.georef = self.georef

        return result

    def drainagebasins(self, outlets=None):
        """Delineate drainage basins from a flow network.

        Parameters
        ----------
        outlets: np.ndarray
            An array containing the linear indices of the outlet nodes
            in column major ('F') order.

        Returns
        -------
        GridObject
            An integer-valued GridObject with a unique label for each drainage
            basin.

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> basins = fd.drainagebasins()
        >>> basins.shufflelabel().plot(cmap="Pastel1",interpolation="nearest")

        """
        if outlets is None:
            basins = np.zeros(self.shape, dtype=np.int64, order=self.order)
            _flow.drainagebasins(basins, self.source, self.target, self.dims)
        else:
            basins = np.zeros(self.shape, dtype=np.uint32, order=self.order)
            indices = self.unravel_index(outlets)
            basins[indices] = np.arange(1, len(outlets) + 1, dtype=np.uint32)
            weights = np.full(self.source.size, 0xffffffff, dtype=np.uint32)
            _stream.traverse_up_u32_or_and(
                basins, weights, self.source, self.target)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = np.array(basins, dtype=np.int64)
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.georef = self.georef

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

        Example
        -------
        >>> dem = tt3.load_dem('bigtujunga')
        >>> fd = tt3.FlowObject(dem)
        >>> print(fd.flowpathextract(12345))
        """
        ch = np.zeros(self.shape, dtype=np.uint32, order=self.order)
        ch[self.unravel_index(idx)] = 1
        edges = np.ones(self.source.size, dtype=np.uint32)
        _stream.traverse_down_u32_or_and(ch, edges, self.source, self.target)

        return self.stream[ch[self.unravel_index(self.stream)] > 0]

    def distance(self):
        """Compute the distance between each node in the flow network

        Returns
        -------
        np.ndarray
            An array (edge attribute list) with the interpixel
            distance. This will be either cellsize or sqrt(2)*cellsize

        Example
        -------
        >>> dem = tt3.load_dem('bigtujunga')
        >>> fd = tt3.FlowObject(dem)
        >>> print(fd.distance())
        """
        d = np.abs(self.source - self.target)
        dist = self.cellsize * np.where(
            (d == self.strides[0]) | (d == self.strides[1]),
            np.float32(1.0),
            np.sqrt(np.float32(2.0)))
        return dist

    def downstream_distance(self) -> GridObject:
        """Calculates the horizontal distance from outlets and ridges
        along the flow network in the downstream direction.

        Returns
        -------
        down_d: GridObject
            A new GridObject containing the distance grid
        """
        # Edge attribute list
        dist = self.distance()

        down_d = np.zeros(self.shape, dtype = np.float32, order=self.order)
        _stream.traverse_down_f32_max_add(down_d, dist, self.source, self.target)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = down_d
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.georef = self.georef

        return result

    def upstream_distance(self) -> GridObject:
        """Calculates the horizontal distance from outlets and ridges
        along the flow network in the upstream direction.

        Returns
        -------
        up_d: GridObject
            A new GridObject containing the distance grid
        """
        # Edge attribute list
        dist = self.distance()

        up_d = np.zeros(self.shape, dtype = np.float32, order=self.order)
        _stream.traverse_up_f32_max_add(up_d, dist, self.source, self.target)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = up_d
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.georef = self.georef

        return result

    def dependencemap(self, l) -> GridObject:
        """Delineate upslope area for specific locations in a DEM.

        Parameters
        -------
        fd: FlowObject
        l: GridObject
            logical grid

        Returns
        -------
        i: GridObject
            logical influence grid (GRIDobj)
        """

        # convert input argument to correct units
        seed = self.ezgetnal(l, dtype = np.uint32)

        # graph traversal algorithm
        i = np.ones(self.source.shape, dtype = np.uint32, order=self.order) # turns on all edges
        _stream.traverse_up_u32_or_and(seed, i, self.source, self.target)

        return l.duplicate_with_new_data(np.asarray(seed))

    def influencemap(self, l) -> GridObject:
        """Delineate downslope area for specific locations in a DEM.

        Parameters
        -------
        fd: FlowObject
        l: GridObject
            logical grid

        Returns
        -------
        i: GridObject
            logical influence grid (GRIDobj)
        """

        # convert input argument to correct units
        seed = self.ezgetnal(l, dtype = np.uint32)

        # graph traversal algorithm
        i = np.ones(self.source.shape, dtype = np.uint32, order=self.order) # turns on all edges
        _stream.traverse_down_u32_or_and(seed, i, self.source, self.target)

        return l.duplicate_with_new_data(np.asarray(seed))

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
