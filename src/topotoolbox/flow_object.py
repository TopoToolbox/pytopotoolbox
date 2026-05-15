"""This module contains the FlowObject class.
"""
from typing import Literal
from dataclasses import dataclass

import numpy as np

# pylint: disable=no-name-in-module
from . import _grid  # type: ignore
from . import _stream  # type: ignore

# pylint: disable=unused-import
from . import _flow # type: ignore
from .grid_object import GridObject
from .interface import validate_alignment

__all__ = ['EdgeSet', 'FlowObject']

@dataclass
class EdgeSet:
    """An unordered collection of weighted, directed edges within a raster"""
    directions: np.ndarray
    scan: np.ndarray
    weights: np.ndarray

    @property
    def count(self):
        """The number of edges in the edgeset"""
        return _flow.edgeset_count(self.directions)

    @property
    def shape(self):
        """The shape of the raster underlying the edgeset"""
        return self.directions.shape

    def merge(self, other):
        """Merge two edgesets"""
        c2 = _flow.edgeset_count_merged(self.directions, other.directions)

        directions = self.directions.copy(order='K')
        scan = np.zeros_like(directions, dtype=np.int64)
        weights = np.zeros(c2, dtype=np.float32)

        _flow.edgeset_merge(weights, scan,
                            directions, self.weights,
                            other.directions, other.weights)

        return EdgeSet(directions, scan, weights)

    def tsort(self):
        """Topologically sort the edges

        Returns
        -------
        (node, source, target, weight)
            - node: The topologically sorted list of pixel indices
            - source: the topologically sorted list of sources of each edge
            - target: the topologically sorted list of targets of each edge
            - weight: the topologically sorted list of weights of each edge
        """
        c = self.count

        stream = np.zeros(self.shape, dtype=np.int64)
        source = np.zeros(c, dtype=np.int64)
        target = np.zeros(c, dtype=np.int64)
        sweight = np.zeros(c, dtype=np.float32)

        stack = np.zeros(self.shape, dtype=np.int64)
        stackdir = np.zeros(self.shape, dtype=np.uint8)
        visited = np.zeros(self.shape, dtype=np.uint8)

        _flow.flow_routing_tsort(stream, source, target,
                                 sweight, stack, stackdir,
                                 self.directions, self.weights, self.scan, visited)

        return (stream, source, target, sweight)

def _d8(dem: np.ndarray) -> EdgeSet:
    """Route flow over the provided DEM with D8
    """
    z = np.asarray(dem, dtype=np.float32)
    directions = np.zeros_like(z, dtype=np.uint8)
    _flow.flow_routing_d8_directions(directions, z)

    scan = np.zeros_like(z, dtype=np.int64)
    c = _flow.edgeset_scan(scan, directions)

    weights = np.zeros(c, dtype=np.float32)
    _flow.flow_routing_d8_weights(weights)

    return EdgeSet(directions, scan, weights)

def _lcat(aux: np.ndarray, demf: np.ndarray, flats: np.ndarray) -> EdgeSet:
    """Route flow over the provided DEM and auxiliary topography using
    least cost auxiliary topography carving
    """
    directions = np.zeros_like(demf, dtype=np.uint8)
    resolved = flats & 1 == 0

    _flow.resolve_flats_lcat(directions, resolved, aux, demf)

    scan = np.zeros_like(demf, dtype=np.int64)
    cr = _flow.edgeset_scan(scan, directions)

    weights = np.zeros(cr, dtype=np.float32)
    _flow.resolve_flats_lcat_weights(weights)

    return EdgeSet(directions, scan, weights)

def _d8_carve(grid: GridObject,
             bc: np.ndarray | GridObject | None = None,
             hybrid: bool = True):
    """Construct a FlowObject using D8 flow routing with least
        cost auxiliary topography carving.

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
    order = (0 if grid.z.flags.f_contiguous else 1)


    (aux, filled_dem, flats) = grid.auxiliary_topography(bc, hybrid)

    node = np.zeros_like(aux, dtype=np.int64)  # node: dtype=np.int64
    direction = np.zeros_like(aux, dtype=np.uint8)
    _grid.flow_routing_d8_carve(node, direction, filled_dem, aux, flats, dims, order)

    # ravel is used here to flatten the arrays. The memory order should not matter
    # because we only need a block of contiguous memory interpreted as a 1D array.
    source = np.zeros(aux.size, dtype=np.int64)  # source: dtype=int64
    target = np.zeros(aux.size, dtype=np.int64)       # target: dtype=int64
    edge_count = _grid.flow_routing_d8_edgelist(source, target, node, direction, dims, order)

    return (direction,
            node,
            source[0:edge_count],
            target[0:edge_count],
            np.ones(edge_count, dtype=np.float32))

class FlowObject():
    """A class containing containing (water-) flow information about a given
    digital elevation model (DEM).
    """

    def __init__(self, grid: GridObject,
                 bc: np.ndarray | GridObject | None = None,
                 method: str = "d8",
                 sink_resolution: str = "carve",
                 **kwargs):
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
        method : str, optional
            The flow routing method to use. Currently supported methods include "d8".
        sink_resolution: str, optional
            The sink resolution method to use. Currently supported
            methods are "carve" and "lcat". The default is "carve".

        Raises
        ------
        ValueError
            The supplied method is not supported.

        Example
        -------
        >>> import topotoolbox
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> flow = topotoolbox.FlowObject(dem)
        """
        if method == "d8":
            if sink_resolution == "carve":
                (direction, stream, source, target, sweight) = _d8_carve(grid, bc, **kwargs)
            elif sink_resolution == "lcat":
                aux, demf, flats = grid.auxiliary_topography(bc)
                e1 = _d8(demf)
                e2 = _lcat(aux, demf, flats)

                em = e1.merge(e2)
                direction = em.directions

                (stream, source, target, sweight) = em.tsort()
            else:
                raise ValueError(f" Sink resolution {method} is not supported")
        else:
            raise ValueError(f"Flow routing {method} is not supported")

        self.path = grid.path
        self.name = grid.name

        self.shape = grid.shape
        self.cellsize = grid.cellsize
        self.strides = tuple(s // grid.z.itemsize for s in grid.z.strides)
        self.order: Literal['F', 'C'] = ('F' if grid.z.flags.f_contiguous
                                       else 'C')

        self.bounds = grid.bounds
        self.transform = grid.transform
        self.georef = grid.georef

        self.direction = direction
        self.stream = stream
        self.source = source
        self.target = target
        self.fraction = sweight

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
        .. plot::

           >>> import topotoolbox
           >>> import matplotlib.pyplot as plt
           >>> dem = topotoolbox.load_dem('bigtujunga')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> _= fd.ezgetnal(dem).plot()
           >>> plt.show()
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
        .. plot::

           >>> import topotoolbox
           >>> import matplotlib.pyplot as plt
           >>> dem = topotoolbox.load_dem('perfectworld')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> acc = fd.flow_accumulation()
           >>> _= acc.plot(cmap='Blues',norm="log")
           >>> plt.show()
        """
        acc = np.array(self.ezgetnal(weights, dtype=np.float32), copy=True, order=self.order)

        fraction = np.ones_like(self.source, dtype=np.float32)

        _stream.traverse_down_f32_add_mul(acc, fraction, self.source, self.target)

        result = GridObject()
        result.path = self.path
        result.name = self.name

        result.z = acc
        result.cellsize = self.cellsize

        result.bounds = self.bounds
        result.transform = self.transform
        result.georef = self.georef

        return result

    def getoutlets(self):
        """Extract outlets from a flow network.

        These are defined as nodes that have no downstream neighbor in
        the network. This includes any internal sinks as well as
        outlets where flow exits the DEM.

        Returns
        -------
        ndarray
            A list of linear indices of outlets. Use
            `FlowObject.unravel_index` to convert to multidimensional
            indices to index into a `GridObject`.

        Example
        -------
        .. plot::

           >>> import topotoolbox
           >>> import matplotlib.pyplot as plt
           >>> from matplotlib.colors import ListedColormap
           >>> dem = topotoolbox.load_dem("bigtujunga")
           >>> fd = topotoolbox.FlowObject(dem)
           >>> outlets = fd.getoutlets()
           >>> j, i = fd.unravel_index(outlets)
           >>> x, y = fd.transform * (i, j)
           >>> _ = dem.plot_hs(cmap=ListedColormap([0.9, 0.9, 0.9]), exaggerate=3)
           >>> _ = plt.scatter(x, y)
           >>> plt.show()
        """
        indegree = np.zeros(self.shape, order=self.order, dtype=np.uint8)
        outdegree = np.zeros(self.shape, order=self.order, dtype=np.uint8)
        _stream.edgelist_degree(indegree, outdegree, self.source, self.target)
        output = (outdegree == 0) & (indegree > 0)

        return np.nonzero(np.ravel(output, order='K'))[0]

    def drainagebasins(self, outlets=None):
        """Delineate drainage basins from a flow network.

        Parameters
        ----------
        outlets: np.ndarray
            An array containing the linear indices of the outlet nodes.

        Returns
        -------
        GridObject
            An integer-valued GridObject with a unique label for each drainage
            basin.

        Example
        -------
        .. plot::

           >>> import topotoolbox
           >>> import matplotlib.pyplot as plt
           >>> dem = topotoolbox.load_dem('perfectworld')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> basins = fd.drainagebasins()
           >>> _= basins.shufflelabel().plot(cmap="Pastel1",interpolation="nearest")
           >>> plt.show()
        """
        if outlets is None:
            outlets = self.getoutlets()

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
        .. plot::

           >>> import topotoolbox
           >>> dem = topotoolbox.load_dem('bigtujunga')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> print(fd.flowpathextract(12345)) # doctest: +SKIP
        """
        ch = np.zeros(self.shape, dtype=np.uint32, order=self.order)
        ch[self.unravel_index(idx)] = 1
        edges = np.ones(self.source.size, dtype=np.uint32)
        _stream.traverse_down_u32_or_and(ch, edges, self.source, self.target)

        return self.stream[ch[self.unravel_index(self.stream)] > 0]

    def node_to_node_distance(self):
        """Compute the distance between each node in the flow network

        Returns
        -------
        np.ndarray
            An array (edge attribute list) with the interpixel
            distance. This will be either cellsize or sqrt(2)*cellsize

        Example
        -------
        .. plot::

           >>> import topotoolbox
           >>> dem = topotoolbox.load_dem('bigtujunga')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> print(fd.node_to_node_distance()) # doctest: +SKIP
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
        dist = self.node_to_node_distance()

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
        dist = self.node_to_node_distance()

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

    def vertdistance2stream(self, stream,
                                      grid: GridObject) -> GridObject:
        """Calculates relative elevation from rivers defined by a stream object.
        Follows the flow paths, not the shortest euclidian distance.
        Also called hand (height above nearest drainage)

        Returns
        -------
        up_z: GridObject
            A new GridObject containing the relative elevation grid
        """

        # pylint: disable=import-outside-toplevel
        # Local import to avoid circular import
        from .stream_object import StreamObject

        if not isinstance(stream, StreamObject):
            raise TypeError('stream must be a StreamObject')

        # Getting hte river location as 2D mask
        mask = stream.gridmask

        # Calculating the relative Z from nodes to their receivers

        ## Elevation for every sources and targets
        zsources = grid.z[self.source_indices]
        ztarget = grid.z[self.target_indices]

        ## dz host the local relief
        dz = zsources - ztarget

        # Masking my rivers (if the source node belongs to the river mask)
        dz[mask[self.source_indices]] = 0.

        ## Summing the global relative elevation to rivers
        up_z = np.zeros(self.shape, dtype = np.float32, order=self.order)
        _stream.traverse_up_f32_max_add(up_z, dz, self.source, self.target)

        return grid.duplicate_with_new_data(up_z)

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
        seed = self.ezgetnal(l, dtype = np.uint8)

        # graph traversal algorithm
        i = np.ones(self.source.shape, dtype = np.uint8, order=self.order) # turns on all edges
        _stream.traverse_up_u8_or_and(seed, i, self.source, self.target)

        return l.duplicate_with_new_data(np.asarray(seed, dtype=np.bool))

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
        seed = self.ezgetnal(l, dtype = np.uint8)

        # graph traversal algorithm
        i = np.ones(self.source.shape, dtype = np.uint8, order=self.order) # turns on all edges
        _stream.traverse_down_u8_or_and(seed, i, self.source, self.target)

        return l.duplicate_with_new_data(np.asarray(seed, dtype=np.bool))

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
