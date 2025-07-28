"""This module contains the StreamObject class.
"""
import math
import warnings
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from shapely.geometry import LineString
import geopandas as gpd
import scipy.sparse as sp

# pylint: disable=no-name-in-module
from clarabel import ZeroConeT, NonnegativeConeT, DefaultSettings, DefaultSolver

from .flow_object import FlowObject
from .grid_object import GridObject
from .interface import validate_alignment
from .stream_functions import imposemin

# pylint: disable=no-name-in-module
from . import _flow  # type: ignore
from . import _stream  # type: ignore

_all_ = ['StreamObject']


class StreamObject():
    """A class to represent stream flow accumulation based on a FlowObject.
    """

    def __init__(self, flow: FlowObject, units: str = 'pixels',
                 threshold: int | float | GridObject | np.ndarray = 0,
                 stream_pixels: GridObject | np.ndarray | None = None,
                 channelheads: np.ndarray | None = None
                 ) -> None:
        """Initializes the StreamObject by processing flow accumulation.

    Parameters
    ----------
    flow : FlowObject
        The input flow object containing source, target, direction, and other
        properties related to flow data.
    units : str, optional
        Units of measurement for the flow data. Can be 'pixels', 'mapunits',
        'm2', or 'km2'. Default is 'pixels'.
    threshold : int | float | GridObject | np.ndarray, optional
        The upslope area threshold for flow accumulation. This can be an
        integer, float, GridObject, or a NumPy array. If more water than in
        the threshold has accumulated in a cell, it is part of the stream.
        Default is 0, which will result in the threshold being generated
        based on this formula: threshold = (avg^2)*0.01
        where shape = (n,m).
    stream_pixels : GridObject | np.ndarray, optional
        A GridObject or np.ndarray made up of zeros and ones to denote where
        the stream is located. Using this will overwrite any use of the
        threshold argument.
    channelheads: (rows, cols), optional
        A tuple of two array-like objects containing the row and
        column indices of the channel heads. All streams downstream of
        the indicated channel heads will be returned in the
        StreamObject.

    Raises
    ------
    ValueError
        If the `units` parameter is not 'pixels', 'mapunits', 'm2', or 'km2'.
    ValueError
        If the shape of the threshold does not match the flow object shape.

    Example
    -------
    >>> dem = topotoolbox.load_dem('perfectworld')
    >>> fd = topotoolbox.FlowObject(dem)
    >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
    >>> plt.subplots()
    >>> dem.plot(cmap="terrain")
    >>> s.plot(color='r')

        """
        if not isinstance(flow, FlowObject):
            err = f"{flow} is not a topotoolbox.FlowObject."
            raise TypeError(err)

        self.cellsize = flow.cellsize
        self.shape = flow.shape
        self.strides = flow.strides
        self._order = flow.order

        # georeference
        self.bounds = flow.bounds
        self.transform = flow.transform
        self.georef = flow.georef

        cell_area = 0.0
        # Calculate the are of a cell based on the units argument.
        if units == 'pixels':
            cell_area = 1.0
        elif units == 'm2':
            cell_area = self.cellsize**2
        elif units == 'km2':
            cell_area = (self.cellsize*0.001)**2
        elif units == 'mapunits':
            if self.georef is not None:
                if self.georef.is_projected:
                    # True so cellsize is in meters
                    cell_area = self.cellsize**2
                else:
                    # False so cellsize is in degrees
                    pass
        else:
            err = (f"Invalid unit '{units}' provided. Expected one of "
                   f"'pixels', 'mapunits', 'm2', 'km2'.")
            raise ValueError(err) from None

        # If stream_pixels are provided, the stream can be generated based
        # on stream_pixels without the need for a threshold
        w = np.zeros(flow.shape, dtype='bool', order='F').ravel(order='K')
        if stream_pixels is not None:
            if not validate_alignment(self, stream_pixels):
                err = (
                    f"stream_pixels shape {stream_pixels.shape}"
                    f" does not match FlowObject shape {self.shape}.")
                raise ValueError(err)

            w = (np.asarray(stream_pixels) != 0).ravel(order='K')

            if threshold != 0:
                warn = ("Since stream_pixels have been provided, the "
                        "input for threshold will be ignored.")
                warnings.warn(warn)
        elif channelheads is not None:
            ch = np.zeros(flow.shape, dtype=np.uint32, order=flow.order)
            ch[channelheads] = 1
            edges = np.ones(flow.source.size, dtype=np.uint32)
            _stream.traverse_down_u32_or_and(
                ch, edges, flow.source, flow.target)
            w = (ch > 0).ravel(order='K')

        # Create the appropriate threshold matrix based on the threshold input.
        else:
            if isinstance(threshold, (int, float)):
                if threshold == 0:
                    avg = (flow.shape[0] + flow.shape[1])//2
                    threshold = np.full(
                        self.shape, math.floor((avg ** 2) * 0.01),
                        dtype=np.float32)
                else:
                    threshold = np.full(
                        self.shape, threshold, dtype=np.float32)
            else:
                if not validate_alignment(self, threshold):
                    err = (
                        f"Threshold GridObject shape {threshold.shape} does "
                        f"not match FlowObject shape: {self.shape}.")
                    raise ValueError(err) from None

                threshold = np.asarray(threshold, dtype=np.float32, order='F')

            # Divide the threshold by how many m^2 or km^2 are in a cell to
            # convert the user input to pixels for further computation.
            threshold /= cell_area

            # Generate the flow accumulation matrix (acc)
            acc = np.zeros(flow.shape, order='F', dtype=np.float32)
            fraction = np.ones_like(flow.source, dtype=np.float32)
            weights = np.ones(flow.shape, order='F', dtype=np.float32)
            _flow.flow_accumulation(
                acc, flow.source, flow.target, fraction, weights, flow.shape)

            # Generate a 1D array that holds all indexes where more water than
            # in the required threshold is collected. (acc >= threshold)
            w = (acc >= threshold).ravel(order='F')

        # Indices of pixels in the stream network
        # This is a node attribute list
        self.stream = np.nonzero(w)[0]

        # Find edges whose source pixel is in the stream network
        u = flow.source
        v = flow.target
        d = flow.direction.ravel(order='F')

        i = w[u]

        # Renumber the nodes of the stream network
        ix = np.zeros_like(w, dtype='int64')
        ix[w] = np.arange(0, self.stream.size)

        # Edges in the stream network
        #
        # Elements of these edge attribute lists are 0-based indices
        # into node attribute lists.
        #
        # To convert these to pixel indices in the original GridObject
        # or FlowObject, use stream[source] or stream[target].
        self.source = ix[u[i]]
        self.target = ix[v[i]]
        self.direction = d[u[i]]

        # misc
        self.path = flow.path
        self.name = flow.name

    @property
    def node_indices(self) -> tuple[np.ndarray, ...]:
        """The row and column indices of the nodes of the stream
        network.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices for the nodes in the stream network. Each of these
        arrays is a node attribute lists and have a length equal to
        the number of nodes in the stream network. This tuple of
        arrays is suitable for indexing GridObjects or arrays shaped
        like the GridObject from which this StreamObject was derived.

        """
        return np.unravel_index(self.stream, self.shape, self._order)

    def node_indices_where(self, nal: np.ndarray) -> tuple[np.ndarray, ...]:
        """The row and column indices of the nodes of the stream
        network where a condition is satisfied.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices for the nodes in the stream network where the input
        Boolean node attribute list `nal` is true. Each of these
        arrays is a node attribute lists and have a length equal to
        the number of nodes in the stream network. This tuple of
        arrays is suitable for indexing GridObjects or arrays shaped
        like the GridObject from which this StreamObject was derived.

        """
        return np.unravel_index(self.stream[nal], self.shape, self._order)

    @property
    def source_indices(self) -> tuple[np.ndarray, ...]:
        """The row and column indices of the sources of each edge in
        the stream network.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices of the sources of each edge in the stream
        network. Each of these arrays is an edge attribute lists and
        have a length equal to the number of edges in the stream
        network. This tuple of arrays is suitable for indexing
        GridObjects or arrays shaped like the GridObject from which
        this StreamObject was derived.

        """
        return np.unravel_index(self.stream[self.source],
                                self.shape, self._order)

    @property
    def target_indices(self) -> tuple[np.ndarray, ...]:
        """The row and column indices of the targets of each edge in
        the stream network.

        Returns
        -------
        tuple of ndarray
            A tuple of arrays containing the row indices and column
        indices of the sources of each edge in the stream
        network. Each of these arrays is an edge attribute lists and
        have a length equal to the number of edges in the stream
        network. This tuple of arrays is suitable for indexing
        GridObjects or arrays shaped like the GridObject from which
        this StreamObject was derived.

        """
        return np.unravel_index(self.stream[self.target],
                                self.shape, self._order)

    def distance(self) -> np.ndarray:
        """
        Compute the pixel-to-pixel distance for each edge.

        Returns
        -------
        np.ndarray, float32
            An edge attribute list with the distance between pixels

        Example
        -------
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> print(s.distance())
        """
        d = np.abs(self.stream[self.source] - self.stream[self.target])

        dist = self.cellsize * np.where(
            (d == self.strides[0]) | (d == self.strides[1]),
            np.float32(1.0),
            np.sqrt(np.float32(2.0)))

        return dist

    def downstream_distance(self) -> np.ndarray:
        """Compute the maximum distance between a node in the stream
        network and the channel head.

        Returns
        -------
        np.ndarray, float32
            A node attribute list with the downstream distances
        """
        d = self.distance()  # Edge attribute list
        dds = np.zeros_like(self.stream, dtype=np.float32)
        _stream.traverse_down_f32_max_add(dds, d, self.source, self.target)

        return dds

    def upstream_distance(self) -> np.ndarray:
        """Compute the upstream distance for a node in the stream network

        Returns
        -------
        np.ndarray, float32
            A node attribute list with the upstream distances
        """
        d = self.distance()
        dds = np.zeros_like(self.stream, dtype=np.float32)
        _stream.traverse_up_f32_max_add(dds, d, self.source, self.target)

        return dds

    def ezgetnal(self, k, dtype=None):
        """Retrieve a node attribute list from k

        Parameters
        ----------
        k : GridObject | np.ndarray | float
            The object from which node values will be extracted. If
            `k` is a `GridObject` or an `np.ndarray` with the same
            shape as the underlying DEM of this `StreamObject`, the
            node values will be extracted from the grid by
            indexing. If `k` is an array with the same shape as the
            node attribute list, `ezgetnal` returns a copy of `k`. If
            `k` is a scalar value, `ezgetnal` returns an array of the
            right shape filled with `k`.

        Returns
        -------
        np.ndarray
            The resulting array will always be a copy of the input
            array.

        Raises
        ------
        ValueError
            If `k` does not have the right shape to be indexed by the
            `StreamObject`.
        """
        if np.isscalar(k):
            nal = np.full(self.stream.shape, k, dtype=None)
        else:
            if validate_alignment(self, k):
                # k is a GridObject or ndarray with the right shape
                # and georeferencing
                # Advanced indexing of k will always return a copy
                nal = k[self.node_indices]

                # We use copy=False in astype to avoid copying that copy
                # if possible
                nal = nal.astype(dtype or nal.dtype, copy=False)
            elif hasattr(k, "shape") and self.stream.shape == k.shape:
                # k is already a node attribute list
                nal = np.array(k, dtype=dtype, copy=True)
            else:
                raise ValueError(f"""{k} is not a node attribute list
                of the appropriate shape.""")

        return nal

    def streampoi(self, point_type: str) -> np.ndarray:
        """Extract points of interest from the stream network

        Currently supported points of interest are 'channelheads',
        'outlets' and 'confluences'

        Parameters
        ----------
        point_type: 'channelheads' or 'outlets' or 'confluences'
            The type of points to select from the stream network

        Returns
        -------
        np.ndarray
            A logical node attribute list indicating the locations of points.

        Raises
        ------
        ValueError
            If an unknown point type is requested.
        """
        indegree = np.zeros(self.stream.size, dtype=np.uint8)
        outdegree = np.zeros(self.stream.size, dtype=np.uint8)
        _stream.edgelist_degree(indegree, outdegree, self.source, self.target)
        if point_type == 'channelheads':
            output = (outdegree > 0) & (indegree == 0)
        elif point_type == 'outlets':
            output = (outdegree == 0) & (indegree > 0)
        elif point_type == 'confluences':
            output = indegree > 1
        else:
            raise ValueError(f"{point_type} is not currently supported")

        return output

    def xy(self, data=None):
        """Compute the x and y coordinates of continuous stream segments

        Arguments
        ---------
        data: tuple, optional
           A tuple of two node attribute lists representing the
           desired x and y values for each pixel in the stream
           network. If this argument is not supplied, the returned x
           and y values are the geographic coordinates of the node.

        Returns
        -------
        list
            A list of lists of (x,y) pairs.
        """
        if data is None:
            # pylint: disable=unbalanced-tuple-unpacking
            j, i = self.node_indices
            xs, ys = self.transform * np.vstack((i, j))
        else:
            xs, ys = data

        vertices = range(self.stream.size)
        edges = range(self.source.size)

        # Construct an adjacency list for the graph,
        # so we can do a depth-first search
        adjacency_list = [[] for _ in vertices]
        for e in edges:
            src = self.source[e]
            tgt = self.target[e]
            adjacency_list[src].append(tgt)

        # Depth-first search of the graph
        visited = np.zeros(self.stream.size, dtype=bool)
        segments = []
        stack = []

        for e in edges:
            src = self.source[e]
            if not visited[src]:
                # Start a new segment
                stack.append(src)
                segments.append([])
            while stack:
                u = stack.pop()
                # Always append the next vertex to the segment
                segments[-1].append((xs[u], ys[u]))
                # If u has already been visited, we stop the segment
                # otherwise, we push its children to visit later
                if not visited[u]:
                    visited[u] = True
                    # Add any neighbors of
                    for v in adjacency_list[u]:
                        stack.append(v)

        return segments

    def to_geodataframe(self):
        '''Convert the stream network to a GeoDataFrame.

        Returns
        ----------
        geopandas.GeoDataFrame
            The GeoDataFrame.

        Example
        -------
        dem = tt3.load_dem('bigtujunga')
        fd = tt3.FlowObject(dem)
        s = tt3.StreamObject(fd)
        s_gdf = s.to_geodataframe()
        '''

        line_geoms = [LineString(coords) for coords in self.xy()]
        gdf = gpd.GeoDataFrame(geometry=line_geoms, crs=self.georef)
        return gdf

    def to_shapefile(self, path: str) -> None:
        '''Convert the stream network to a georeferenced shapefile.

        Parameters
        ----------
        path : str
             path where the shapefile will be saved.

        Example
        -------
        dem = tt3.load_dem('bigtujunga')
        fd = tt3.FlowObject(dem)
        s = tt3.StreamObject(fd)
        s.to_shapefile('stream_network.shp')
        '''
        gdf = self.to_geodataframe()
        gdf.to_file(path)

    def plot(self, ax=None, scalex=True, scaley=True, **kwargs):
        """Plot the StreamObject

        Stream segments as computed by StreamObject.xy are plotted
        using a LineCollection.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the StreamObject. If no axes are
            given, the current axes are used.

        scalex: bool, optional
            Autoscale the x-axis limits. Defaults to `True`. If the
            x-axis limits have been set manually with `set_xlim`, the
            autoscaling will not be applied regardless of the value of
            `scalex`.

        scaley: bool, optional
            Autoscale the y-axis limits. Defaults to `True`. If the
            y-axis limits have been set manually with `set_ylim`, the
            autoscaling will not be applied regardless of the value of
            `scaley`.

        **kwargs
            Additional keyword arguments are forwarded to LineCollection

        Returns
        -------
        matplotlib.axes.Axes
            The axes into which the StreamObject has been plotted.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> plt.subplots()
        >>> dem.plot(cmap="terrain")
        >>> s.plot(color='r')
        """

        if ax is None:
            ax = plt.gca()
        collection = LineCollection(self.xy(), **kwargs)
        ax.add_collection(collection)
        ax.autoscale_view(scalex=scalex, scaley=scaley)
        return ax

    def plotdz(self, z, ax=None, dunit: str = 'm', doffset: float = 0,
               scalex=True, scaley=True, **kwargs):
        """Plot a node attribute list against upstream distance

        Parameters
        ----------
        z: GridObject, np.ndarray
          The node attribute list that will be plotted

        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the StreamObject. If no axes are
            given, the current axes are used.

        dunit: str, optional
            The unit to plot the upstream distance. Should be either
            'm' for meters or 'km' for kilometers.

        doffset: float, optional
            An offset to be applied to the upstream distance.
            `doffset` should be in the units specified by `dunit`.

        scalex: bool, optional
            Autoscale the x-axis limits. Defaults to `True`. If the
            x-axis limits have been set manually with `set_xlim`, the
            autoscaling will not be applied regardless of the value of
            `scalex`.

        scaley: bool, optional
            Autoscale the y-axis limits. Defaults to `True`. If the
            y-axis limits have been set manually with `set_ylim`, the
            autoscaling will not be applied regardless of the value of
            `scaley`.

        **kwargs
            Additional keyword arguments are forwarded to LineCollection

        Returns
        -------
        matplotlib.axes.Axes
            The axes into which the plot as been added

        Raises
        ------
        ValueError
            If `dunit` is not one of 'm' or 'km'.

        """

        if ax is None:
            ax = plt.gca()
        z = self.ezgetnal(z)
        dist = np.zeros_like(z, dtype=np.float32)
        a = np.ones_like(z, dtype=np.float32)

        # Compute upstream distance using streamquad_trapz_f32
        # Another traversal might be more efficient in the future
        _stream.streamquad_trapz_f32(dist, a,
                                     self.source,
                                     self.target,
                                     self.distance())

        if dunit == 'km':
            dist /= 1000
        elif dunit != 'm':
            raise ValueError("dunit must be one of 'm' or 'km'")

        dist += doffset

        collection = LineCollection(self.xy((dist, z)), **kwargs)
        ax.add_collection(collection)
        ax.autoscale_view(scalex=scalex, scaley=scaley)
        return ax

    def chitransform(self,
                     upstream_area: GridObject | np.ndarray,
                     a0: float = 1e6,
                     mn: float = 0.45,
                     k: GridObject | np.ndarray | None = None,
                     correctcellsize: bool = True):
        """Coordinate transformation using the integral approach

        Transforms the horizontal spatial coordinates of a river
        longitudinal profile using an integration in upstream
        direction of drainage area (chi, see Perron and Royden, 2013).

        Parameters
        ----------
        upstream_area : GridObject | np.ndarray
            Raster with the upstream areas. Must be the same size and
            projection as the GridObject used to create the
            StreamObject.
        a0 : float, optional
            Reference area in the same units as the upstream_area
            raster. Defaults to 100_000.
        mn : float, optional
            mn-ratio. Defaults to 0.45.
        k  : GridObject | np.ndarray | None, optional
            Erosional efficiency, which may vary spatially. If `k` is
            supplied, then `chitransform` returns the time needed for
            a signal (knickpoint) propagating upstream from the outlet
            of the stream network. If `k` has units of m^(1 - 2m) / y,
            then time will have units of y. Note that calculating the
            response time requires the assumption that n=1. Defaults
            to None, which does not use the erosional efficiency.
        correctcellsize : bool, optional
            If true, multiplies the `upstream_area` raster by
            `self.cellsize**2`. Use if `a0` has the same units of
            `self.cellsize**2` and `upstream_area` has units of
            pixels, such as the default output from
            `flow_accumulation`. If the units of `upstream_area` are
            already m^2, then set correctcellsize to False. Defaults
            to True.

        Raises
        ------
        ValueError
            If `upstream_area` or `k` does not have the right shape to
            be indexed by the `StreamObject`.
        TypeError
            If `upstream_area` or `k` does not represent a type of data that can be
            extracted into a node attribute list.
        TypeError
            If the modified upstream area is not a supported floating point type.
        """

        # Retrieve node attribute lists
        a = self.ezgetnal(upstream_area)

        if correctcellsize:
            a = a * self.cellsize**2

        # Set up k
        if k is not None:
            node_k = self.ezgetnal(k)
            a = (1 / node_k) * (1 / a)**mn
        else:
            a = (a0 / a)**mn

        # Cumulative trapezoidal integration
        weight = self.distance()
        c = np.zeros_like(a)
        if a.dtype == np.float32:
            _stream.streamquad_trapz_f32(c,
                                         a,
                                         self.source,
                                         self.target,
                                         weight)
        elif a.dtype == np.float64:
            _stream.streamquad_trapz_f64(c,
                                         a,
                                         self.source,
                                         self.target,
                                         weight)
        else:
            # This is probably unreachable
            raise TypeError("modified area is not a floating point object")

        return c

    def trunk(self, downstream_distance: np.ndarray | None = None,
              flow_accumulation: GridObject | None = None) -> 'StreamObject':
        """Reduces a stream network to the longest streams in each stream
        network tree (e.g. connected component). The algorithm identifies
        the main trunk by sequently tracing the maximum downstream
        distance in upstream direction.

        Parameters
        ----------
        flow_accumulation : Gridobject, optional
            A GridObject filled with flow accumulation values (as returned by
            the function FlowObject.flow_accumulation). Defaults to None.
        downstream_distance : np.ndarray, optional
            A numpy ndarray node-attribute list as generated by ezgetnal().
            This argument overwrites the flow_accumulation if used.
            Defaults to None.

        Returns
        -------
        StreamObject
            StreamObject with truncated streams.

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> s2 = s.klargestconncomps(1)
        >>> st = s2.trunk()
        >>> fig,ax = plt.subplots()
        >>> dem.plot(ax=ax,cmap="terrain")
        >>> s.plot(ax=ax, color='r')
        >>> s2.plot(ax=ax,color='k')
        >>> st.plot(ax=ax, color='b')
        """

        stream_network_size = len(self.stream)

        if not downstream_distance is None:
            pass
        elif not flow_accumulation is None:
            downstream_distance = self.ezgetnal(flow_accumulation)
        else:
            downstream_distance = self.downstream_distance()

        sparse_distance = csr_matrix(
            (downstream_distance[self.source] + 1, (self.source, self.target)),
            shape=(stream_network_size, stream_network_size))

        # Identify outlet reaches
        any_column = np.array(sparse_distance.sum(axis=0) > 0).flatten()
        any_row = np.array(sparse_distance.sum(axis=1) > 0).flatten()
        outlets = any_column & ~any_row

        trunks_max = np.argmax(sparse_distance, axis=0)

        max_neighbor = np.zeros(stream_network_size, dtype=bool)
        max_neighbor[trunks_max] = True

        trunks = np.zeros(stream_network_size, dtype=bool)
        trunks[outlets] = True

        for r in range(len(self.source) - 1, -1, -1):
            trunks[self.source[r]] = trunks[self.target[r]
                                            ] and max_neighbor[self.source[r]]

        result = self.subgraph(trunks)
        return result

    def conncomps(self) -> np.ndarray:
        """Label connected components in a stream network

        Returns
        -------
        np.ndarray
           A node attribute list containing the labels of each
           connected component. The labels range from 1 to the number
           of connected components in the network.
        """
        outlets = self.streampoi('outlets')

        labels = np.zeros(self.stream.size, dtype=np.int64)
        labels[outlets] = np.arange(np.count_nonzero(outlets)) + 1

        _stream.propagatevaluesupstream_i64(labels, self.source, self.target)

        return labels

    def klargestconncomps(self, k=1) -> 'StreamObject':
        """Extract the k largest connected components of the stream network

        Components are ordered by the number of stream network pixels.

        Parameters
        ----------
        k : integer, optional
            The number of components to keep. The default is 1

        Returns
        -------
        StreamObject
            A new StreamObject containing only the k largest connected components

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> s2 = s.klargestconncomps(1)
        >>> fig, ax = plt.subplots()
        >>> dem.plot(ax=ax,cmap="terrain")
        >>> s2.plot(ax=ax,color='k')
        >>> plt.show()
        """
        nv = self.stream.size
        ne = self.source.size

        outlets = self.streampoi('outlets')

        # Count the nodes in each connected component of the stream
        # network.
        # This might be slightly inconsistent with the MATLAB
        # implementation which sorts by the number of edges in each
        # connected component.
        acc = np.ones(nv, dtype=np.float32)
        weights = np.ones(ne, dtype=np.float32)
        _stream.traverse_down_f32_add_mul(
            acc, weights, self.source, self.target)

        # Indices of the outlets in a node attribute list
        outlet_indices = np.flatnonzero(outlets)

        # Indices of the sorted accumulation values, from lowest to highest
        ixs = np.argsort(acc[outlets])

        # conncomps will be 1 for all pixels in the k largest
        # connected components
        conncomps = np.zeros(nv, dtype=np.uint8)

        # Initialize to 1 at the outlets of the k largest components
        conncomps[outlet_indices[ixs[-k:]]] = 1

        # And propagate those values from the outlets upstream
        _stream.propagatevaluesupstream_u8(conncomps, self.source, self.target)

        # Convert to boolean array so we can index with it
        conncomps_mask = conncomps > 0

        result = self.subgraph(conncomps_mask)
        return result

    def subgraph(self, nal):
        """Extract a subset of the stream network

        Parameters
        ----------
        nal: GridObject or np.ndarray
            A logical node attribute list indicating the desired
            nodes of the new stream network

        Returns
        -------
        StreamObject
            A StreamObject representing the desired subset of the
            stream network.

        Example
        -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> shape = dem.shape
        >>> arr = (np.arange(np.prod(shape))<np.prod(shape)//4).reshape(shape)
        >>> s2 = s.subgraph(arr)
        >>> fig,ax = plt.subplots()
        >>> dem.plot(ax=ax,cmap="terrain")
        >>> s2.plot(ax=ax,color='k')
        """

        nal = self.ezgetnal(nal)
        nal = nal > 0
        result = copy.copy(self)

        result.stream = self.stream[nal]

        new_indices = np.cumsum(nal) - 1

        valid_edges = nal[self.source] & nal[self.target]

        result.source = self.source[valid_edges]
        result.target = self.target[valid_edges]

        result.source = new_indices[result.source]
        result.target = new_indices[result.target]

        # MATLAB cleans the result, but this leads to a circular
        # dependency between `subgraph` and `clean` that confuses
        # things.

        # TODO(wkearn): return indices into the original node
        # attribute list
        return result

    def clean(self) -> 'StreamObject':
        """Remove disconnected nodes in stream networks

        Returns
        -------
        StreamObject
            A stream network where all isolated nodes have been removed

        Example
        -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('bigtujunga')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000)
        >>> sc = s.clean()
        >>> assert sc.stream.shape <= s.stream.shape
        """

        indegree = np.zeros(self.stream.size, dtype=np.uint8)
        outdegree = np.zeros(self.stream.size, dtype=np.uint8)
        _stream.edgelist_degree(indegree, outdegree, self.source, self.target)
        nal = (indegree != 0) | (outdegree != 0)

        return self.subgraph(nal)

    def upstreamto(self, nodes) -> 'StreamObject':
        """Extract the portion of the stream network upstream of the given nodes

        Parameters
        ----------
        nodes: GridObject or np.ndarray
            A logical node attribute list or grid that is True for the desired nodes.

        Returns
        -------
        StreamObject
            A stream network containing those nodes of the original
            one that are upstream of the given nodes.

        Example
        -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> confluences = s.streampoi('confluences')
        >>> s2 = s.upstreamto(confluences)
        >>> fig,ax = plt.subplots()
        >>> dem.plot(ax=ax,cmap="terrain")
        >>> s2.plot(ax=ax,color='k')
        """
        nal = self.ezgetnal(nodes, dtype=np.uint32)

        edges = np.ones(self.source.size, dtype=np.uint32)
        _stream.traverse_up_u32_or_and(nal, edges, self.source, self.target)

        return self.subgraph(nal)

    def downstreamto(self, nodes) -> 'StreamObject':
        """Extract the portion of the stream network downstream of the given nodes

        Parameters
        ----------
        nodes: GridObject or np.ndarray
            A logical node attribute list or grid that is True for the desired nodes.

        Returns
        -------
        StreamObject
            A stream network containing those nodes of the original
            one that are downstream of the given nodes.

        Example
        -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('perfectworld')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd,threshold=1000,units='pixels')
        >>> confluences = s.streampoi('confluences')
        >>> s2 = s.downstream(confluences)
        >>> fig,ax = plt.subplots()
        >>> dem.plot(ax=ax,cmap="terrain")
        >>> s2.plot(ax=ax,color='k')
        """
        nal = self.ezgetnal(nodes, dtype=np.uint32)

        edges = np.ones(self.source.size, dtype=np.uint32)
        _stream.traverse_down_u32_or_and(nal, edges, self.source, self.target)

        return self.subgraph(nal)

    def gradient(self, dem, impose=False) -> 'np.ndarray':
        """Calculates the stream slope for each node in the stream
        network S based on the associated digital elevation model DEM.

        Parameters
        ----------
        dem: GridObject or np.ndarray
            A node attribute list or grid that provides the elevation that we take the gradient of.
        impose: bool
            Minima imposition to avoid negative slopes (see imposemin)

        Returns
        -------
        s
            stream gradient as node-attribute list
        """

        # get node attribute list with elevation values
        z = self.ezgetnal(dem)

        if impose:
            z = imposemin(self, z)

        # inter-node distance
        d = self.distance()

        # forward case
        s = np.zeros(self.stream.size)
        s[self.source] = (z[self.source]-z[self.target])/d

        return s

    def ksn(self, dem, a, impose=False, theta=0.45) -> 'np.ndarray':
        """Returns the normalized steepness index using a default concavity
        index of 0.45.

        Parameters
        ----------
        dem: GridObject or np.ndarray
            Digital elevation model
        a: GridObject
            Flow accumulation as returned by flowacc (GridObject)
        impose: bool
            Minima imposition to avoid negative slopes (see imposemin
        theta: float
            Concavity (default 0.45)

        Returns
        -------
        k
            Normalized steepness index
        """
        z = self.ezgetnal(dem)
        a = self.ezgetnal(a)

        # minima imposition to avoid negative gradients
        if impose:
            z = imposemin(self, z)

        # calculate gradient
        g = self.gradient(z)

        # upslope area
        a = a*self.cellsize**2

        # calculate k
        k = g/(a**(-theta))

        return k

    def streamorder(self, method='strahler') -> np.ndarray:
        """Calculates stream order from the StreamObject using the Strahler
        or Shreve method

        Parameters
        ----------
        s: StreamObject
        method: string
            Strahler (default) or Shreve method

        Returns
        -------
        s_order
            Stream order for each node in StreamObject
        """

        s_order = self.streampoi('channelheads')
        s_order = s_order.astype(dtype=np.float32)
        w = np.ones(self.source.shape, dtype=np.float32)

        if method.lower() == 'strahler':
            _stream.traverse_down_f32_strahler(
                s_order, w, self.source, self.target)
        elif method.lower() == 'shreve':
            _stream.traverse_down_f32_add_mul(
                s_order, w, self.source, self.target)
        else:
            raise ValueError("Invalid type. Choose 'strahler' or 'shreve'.")

        return s_order

    def crslin(self, dem, k = 1, mingradient=0.0, attachheads=False, attachtomin=False):
        """ Elevation values along stream networks are frequently affected by
        large scatter, often as a result of data artifacts or errors. This
        function returns a node attribute list of elevations calculated by
        regularized smoothing.

        The algorithm written in this function follows Appendix A2 in the Schwanghart
        and Scherler 2017 paper.

        Parameters:
        ----------
        s: StreamObject
        dem: GridObject | np.ndarray
            DEM
        k: float
            positive scalar that dictates the degree of stiffness. Default is 1
        mingradient: double
            Minimum downward gradient.
            Choose carefully, because length profile may dip to steeply.
            Set this parameter to nan if you do not wish to have a monotonous
            dowstream elevation decrease.
        attachtomin: bool
            Smoothed elevations will not exceed local minima along the
            downstream path. (only applicable if 'mingradient' is not nan)
        attachheads: bool
            If true, elevations of channelheads are fixed. (only applicable
            if 'mingradient' is not nan). Note that for large K, setting
            attachheads to true can result in excessive smoothing and
            underestimation of elevation values directly downstream to channelheads.

        Returns
        ----------
        zs:
            node attribute list with smoothed elevation values
        """

        # get node attribute list with elevation values
        z = self.ezgetnal(dem, dtype='double')  # elevation values of the dem
        nr = z.size
        ne = self.source.size # num of edges

        if any(np.isnan(z)):
            raise ValueError('DEM or z may not contain any NaNs')

        # CRS linear algorithm

        # Clarabel only accepts csc arrays
        # Identity Matrix
        identity_matrix = sp.eye_array(nr).tocsc()

        #########################################
        # Compute second derivative matrix (C in equation A5)
        # find upstream and downstream nodes
        ix = self.source  # upstream
        ixc = self.target  # downstraeam

        # boolean array to store nodes that are both sources and targets
        i = np.isin(ixc, ix)
        # creates a dictionary to store ix
        dic = dict(zip(ix, np.arange(self.source.shape[0])))
        # values as key and their indicies as values
        keys = np.array([dic[j] for j in ixc[i]])

        # [i-1 (downstream), i, i+1 (upstream)]
        colix = np.array([ixc[keys], ixc[i], ix[i]]).T
        nrrows = colix.shape[0]
        rowix = np.tile(np.arange(nrrows).reshape(-1, 1), 3)

        # compute distance values between nodes
        d = self.upstream_distance()

        xj = d[colix[:, 0]]  # downstream node of i
        xi = d[colix[:, 1]]
        xk = d[colix[:, 2]]  # upstream node of i

        # Dense C matrix (equation A4). Must be converted to kvxopt spare matrix for qp
        values = np.array(
            [2/((xi-xj)*(xk-xj)), -2/((xk-xi)*(xi-xj)), 2/((xk-xi)*(xk-xj))])

        # Second derivative matrix (C)
        c = sp.csc_array((values.T.flatten(), (rowix.flatten(), colix.flatten())), (nrrows, nr))

        #########################################
        # Compute s parameter (equation A7)
        delta_x = self.cellsize  # spatial resolution
        n = nr  # number of data points
        p = nrrows  # num. of second derivative equations = num. of nodes where second derivative
        # can be computed (number of i nodes --> with both upstream and dowsntream neighbours)
        s_parameter = ((delta_x)**2)*k*math.sqrt(n/p)

        #########################################
        # Compute parameters for quad programming
        # Compute matrix A and vector b (equation A9)
        a_2 = s_parameter*c  # C*s
        a_matrix = sp.vstack([identity_matrix, a_2], format = 'csc')
        b = np.concatenate([z, np.zeros(nrrows)])

        # Compute function parameters for quadratic programming (equation A11)
        f = -2*(a_matrix.T @ b)
        h_matrix = 2*(a_matrix.T @ a_matrix).tocsc()

        #########################################
        # Constraints
        # Equivalent constraint. (second constr. in equation A11)
        if attachheads:
            channelheads = self.streampoi('channelheads')
            nc = np.count_nonzero(channelheads)  # number of channelheads
            # indices of channelheads positions in array
            ids = np.where(channelheads)

            # matrix with the channelheads on main diagonal
            i_eq = sp.csc_array((channelheads[channelheads].astype(float),
                                  (np.arange(nc), ids[0])), shape=(nc, nr))
            z_eq = z[channelheads]

        else:
            i_eq = sp.csc_array((0, nr))
            z_eq = np.zeros(0)

        # Inequality constraints
        # Gradient constraint
        delta = np.array(1/(d[self.source]-d[self.target]))  # cellsize

        gradient = sp.csc_array(
            (delta, (np.arange(ne), self.target)),shape=(ne, nr)) - sp.csc_array(
                (delta, (np.arange(ne), self.source)), shape=(ne, nr))

        g_min = np.full(ne, mingradient)

        # Set up matrix G and vector g_min in equation A11. Here they are defined as M and h
        # M and h contain all inequality constraints, including upperbound (attachtomin).
        if attachtomin:
            # matrix with inequality constraints [gradient, upper bound]
            m_matrix = sp.vstack([gradient, identity_matrix], format = 'csc')
            h = np.concatenate([-g_min, z])
        else:
            m_matrix = gradient
            h = -g_min

        m_matrix_2 = sp.vstack([i_eq, m_matrix]).tocsc()
        h = np.concatenate([z_eq, h])

        # Solve quadratic programming with the constraints
        cones = [ZeroConeT(i_eq.shape[0]), NonnegativeConeT(m_matrix.shape[0])]

        settings = DefaultSettings()
        settings.verbose = False

        solver = DefaultSolver(h_matrix, f, m_matrix_2, h, cones, settings)
        sol = solver.solve()

        zs = np.array(sol.x[0:nr])

        return zs

    def quantcarve(self, dem, tau = 0.5, mingradient = 0.0, fixedoutlet = False) -> np.ndarray:
        """
        Elevation values along stream networks are frequently affected by
        large scatter, often as a result of data artifacts or errors. This
        function returns a node attribute list of elevations calculated by
        carving the DEM. Conversely to conventional carving, quantcarve will
        not run along minimas of the DEM. Instead, quantcarve returns a
        profile that runs along the tau's quantile of elevation conditional
        horizontal distance of the river profile.

        The algorithm written in this function follows Appendix A3 in the Schwanghart
        and Scherler 2017 paper.

        Parameters
        ----------
        s: StreamObject
        dem: GridObject | np.ndarray
            Digital Elevation Model
        tau: float
            Quantile. Default is 0.5
        mingradient: positive scalar
            Minimum downward gradient.
        fixedoutlet: bool
            If true, elevations of outlets are fixed.
        mingradient: float
            Minimum downward gradient.

        Returns
        -------
        zs
            Node attribute list with with smoothed elevation values
        """

        # get node attribute list with elevation values
        z = self.ezgetnal(dem, dtype = 'double') # elevation values of the dem
        nr = z.size
        ne = self.source.size # number of edges

        # Quantcarve algorithm (equation A12)
        d = self.upstream_distance() # had to use downstream distance
        f = np.vstack((tau*np.ones((nr, 1)), (1-tau)*np.ones((nr,1)), np.zeros((nr,1))))
        f = f.flatten()

        identity_matrix = sp.eye(nr).tocsc()
        zero_matrix = sp.csc_array((nr, nr))
        zero_edge_matrix = sp.csc_array((ne, nr))

        #########################################
        # Constraints
        # gradient
        delta = np.array(1/(d[self.source]-d[self.target])) # cellsize
        gradient = gradient = sp.csc_array(
            (delta, (np.arange(ne), self.target)), shape=(ne, nr)) - sp.csc_array(
                (delta, (np.arange(ne), self.source)), shape=(ne, nr))

        h_matrix = sp.csc_array((3*nr,3*nr)).tocsc() # set quadratic part of the problem
                                                     # to 0 to make it linear

        b = mingradient * np.ones(ne) # min gradient

        # bounds constraints
        lb = np.concatenate([np.zeros(nr), np.zeros(nr)])

        # stacked inequality constraints
        m_matrix = sp.block_array(
                    [[zero_edge_matrix, zero_edge_matrix, gradient],
                    [-identity_matrix, zero_matrix, zero_matrix],
                    [zero_matrix, -identity_matrix, zero_matrix]
                    ])
        h = np.concatenate([-b, lb])

        # equality constraint
        if fixedoutlet:
            outlet = self.streampoi('outlets')
            p = sp.diags_array((~outlet).astype(int), offsets = 0, shape = (nr,nr), format='csc')
            a_eq = sp.block_array([[p, -p, identity_matrix]]).tocsc()
        else:
            a_eq = sp.block_array([[identity_matrix, -identity_matrix, identity_matrix]]).tocsc()

        b_eq = np.array(z, copy = True)

        # All constraints stacked
        m_matrix_2 = sp.vstack([a_eq, m_matrix], format = 'csc')
        h_2 = np.concatenate([b_eq, h])

        #########################################
        # solve linear programming problem
        cones = [ZeroConeT(nr), NonnegativeConeT(ne + 2 * nr)]

        settings = DefaultSettings()
        settings.verbose = True

        solver = DefaultSolver(h_matrix, f, m_matrix_2, h_2, cones, settings)
        sol = solver.solve()

        #us = np.array(sol.x[0:nr])
        #vs = np.array(sol.x[nr:2*nr])
        zs = np.array(sol.x[2*nr:3*nr])

        return zs

    def crs(self, dem, tau = 0.5, k = 1, mingradient = 0.0, fixedoutlet = False) -> np.ndarray:
        """
        Elevation values along stream networks are frequently affected by
        large scatter, often as a result of data artifacts or errors. This
        function returns a node attribute list of smoothed elevation values
        calculated by nonparametric quantile regression with monotonicity
        constraints.

        The algorithm written in this function follows Appendix A4 in the Schwanghart
        and Scherler 2017 paper.

        Parameters:
        ----------
        s: StreamObject
        dem: GridObject | np.ndarray
            DEM
        tau: float
            Quantile. Default is 0.5
        k: float
            positive scalar that dictates the degree of stiffness. Default is 1
        mingradient: double
            Minimum downward gradient.
            Choose carefully, because length profile may dip to steeply.
            Set this parameter to nan if you do not wish to have a monotonous
            dowstream elevation decrease.
        fixedoutlet: bool
            If true, elevations of outlets are fixed.

        Returns
        ----------
        zs:
            node attribute list with smoothed elevation values
        """

        # get node attribute list with elevation values
        z = self.ezgetnal(dem, dtype=np.float64)  # node_data[:,0]
        if any(np.isnan(z)):
            raise ValueError('DEM or z may not contain any NaNs')

        nr = z.size # num. of nodes
        ne = self.source.size # num of edges

        # a special case. The stream network needs at least three nodes in a row. If
        # there are less, second derivative matrix is empty, and crs uses quantcarve instead
        if nr < 3:
            print('nr < 3')
            return self.quantcarve(dem, tau, mingradient, fixedoutlet='True')

        # CRS Algorithm
        # Identity Matrix & zero matrices
        identity_matrix = sp.eye_array(nr)
        zero_matrix = sp.csc_array((nr, nr))
        zero_edge_matrix = sp.csc_array((ne, nr))

        #########################################
        # Compute Second Derivative Matrix (C in equation A5)
        # find upstream and downstream nodes
        ix = self.source
        ixc = self.target

        # boolean array to store nodes that are both sources and targets
        i = np.isin(ixc, ix)
        dic = dict(zip(ix, np.arange(self.source.shape[0]))) # dictionary to store ix
        keys = np.array([dic[j] for j in ixc[i]]) # values as key and their indicies as values

        # [i-1 (downstream), i, i+1 (upstream)]
        colix = np.array([ixc[keys], ixc[i], ix[i]]).T
        nrrows = colix.shape[0]
        rowix = np.tile(np.arange(nrrows).reshape(-1, 1), 3)

        d = self.upstream_distance()

        xj = d[colix[:, 0]] # downstream node of i
        xi = d[colix[:, 1]]
        xk = d[colix[:, 2]] # upstream node of i

        # C matrix (equation A4)
        values = np.array([2/((xi - xj) * (xk - xj)), -2 /
                        ((xk-xi)*(xi-xj)), 2/((xk-xi)*(xk - xj))]).T
        c = sp.csc_array(
            (values.flatten(), (rowix.flatten(), colix.flatten())), (nrrows, nr))

        #########################################
        # Compute s parameter (equation A7)
        delta_x = self.cellsize
        n = nr
        p = nrrows
        s_parameter = ((delta_x)**2) * k * np.sqrt(n/p)

        #########################################
        # Compute function parameters for quadratic part of the problem
        # Compute matrix D (equation A15)
        d_2 = s_parameter*c
        d_matrix = sp.block_array([[zero_matrix], [d_2]])
        b_matrix = 2*(d_matrix.T @ d_matrix)
        h_matrix = sp.block_diag(
                [zero_matrix, zero_matrix, b_matrix], format='csc')

        # Compute function parameters for linear part of the problem
        f = np.concatenate((tau*np.ones(nr), (1-tau)*np.ones(nr), np.zeros(nr)))

        #########################################
        # Constraints
        # Gradient
        delta = 1/(d[self.source] - d[self.target])

        gradient = sp.csc_array((delta, (np.arange(ne), self.target)), shape=(
            ne, nr)) - sp.csc_array((delta, (np.arange(ne), self.source)), shape=(ne, nr))

        b = mingradient * np.ones(ne) # min gradient

        # Bounds
        lb = np.concatenate([np.zeros(nr), np.zeros(nr)])

        # Inequality Constraints
        m_matrix = sp.block_array(
            [[zero_edge_matrix, zero_edge_matrix, gradient],
            [-identity_matrix, zero_matrix, zero_matrix],
            [zero_matrix, -identity_matrix, zero_matrix]
            ])

        h = np.concatenate([-b, lb])

        # Equality Constraint
        if fixedoutlet:
            outlet = self.streampoi('outlets')
            p = sp.diags_array((~outlet).astype(np.int64), offsets = 0,
                               shape = (nr,nr), format='csc')
            a_eq = sp.block_array([[p, -p, identity_matrix]])
        else:
            a_eq = sp.block_array(
                [[identity_matrix, -identity_matrix, identity_matrix]])

        b_eq = np.array(z, copy=True)

        a_matrix = sp.vstack([a_eq, m_matrix], format='csc')
        g = np.concatenate([b_eq, h])

        cones = [ZeroConeT(nr), NonnegativeConeT(ne + 2 * nr)]

        settings = DefaultSettings()
        settings.verbose = False

        #########################################
        # Compute algorithm solution
        solver = DefaultSolver(h_matrix, f, a_matrix, g, cones, settings)
        sol = solver.solve()

        #us = np.array(sol.x[0:nr])
        #vs = np.array(sol.x[nr:2*nr])
        zs = np.array(sol.x[2*nr:3*nr])

        return zs

    def lowerenv(self, dem: GridObject | np.ndarray, kn: np.ndarray) -> np.ndarray:
        """Compute the lower convex envelope of a stream profile

        Parameters
        ----------
        dem: GridObject or np.ndarray

            The elevation data of the stream network provided either
            as a GridObject compatible with this StreamObject or as a
            node attribute list. The elevation data should be
            decreasing downstream. Preprocess with imposemin or
            quantcarve if necessary.

        kn: np.ndarray
            A logical node attribute list that is true for nodes that
            should be considered knickpoints, where the stream profile
            need not be convex.

        Returns
        -------
        np.ndarray
            A node attribute list containing the elevation of the
            smoothed profile.

        Example
        -------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('bigtujunga')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd)
        >>> s = s.klargestconncomps(1)
        >>> z = topotoolbox.imposemin(s, dem)
        >>> kn = np.zeros(len(z), dtype=bool)
        >>> ze = s.lowerenv(z, kn)
        >>> fig,ax = plt.subplots()
        >>> s.plotdz(dem, ax=ax, color='gray')
        >>> s.plotdz(ze, ax=ax, color='black')

        """
        z = self.ezgetnal(dem, dtype=np.float32)
        d = self.upstream_distance()

        nv = len(z)
        ix = np.zeros(nv, dtype=np.int64)
        onenvelope = np.zeros(nv, dtype=np.uint8)

        _stream.lowerenv(z, kn.astype(np.uint8),
                         d, ix, onenvelope,
                         self.source, self.target)

        return z

    def knickpointfinder(self, dem: GridObject | np.ndarray,
                         knickpoints: np.ndarray | None = None,
                         tolerance: float = 100.0,
                         iterations: int | None = None) -> np.ndarray:
        """Find knickpoints in river profiles

        Rivers that adjust to changing base levels or have diverse
        lithologies often feature knickzones, i.e. pronounced convex
        sections that separate the otherwise concave equilibrium
        profile. The profile should be monotoneously decreasing (see
        imposemin, quantcarve, crs).

        This function extracts knickpoints, i.e. sharp convex sections
        in the river profile. This is accomplished by an algorithm
        that adjusts a strictly concave upward profile to an actual
        profile in the DEM or node-attribute list. The algorithm
        iteratively relaxes the concavity constraint at those nodes in
        the river profile that have the largest elevation offsets
        between the strictly concave and actual profile until the
        offset falls below a user-defined tolerance.

        Parameters
        ----------
        dem: GridObject or np.ndarray

            The elevation data of the stream network provided either
            as a GridObject compatible with this StreamObject or as a
            node attribute list. The elevation data is automatically
            processed using imposemin to ensure that it is decreasing
            downstream. If additional smoothing of the stream profile
            is needed, use a function like quantcarve or crslin and
            pass the resulting node attribute list to
            `knickpointfinder`.

        knickpoints: np.ndarray

            A logical node attribute list that is True for any stream
            network nodes that should be considered knickpoints a priori

        tolerance: float

            The maximum difference between the DEM and a modeled
            convex profile that will be considered a
            knickpoint

            Setting this higher will identify fewer knickpoints while
            setting it lower finds more knickpoints. It is set by
            default to 100 m, but users should choose a tolerance
            based on their data and needs.

        iterations: int

            The maximum number of iterations that will be
            run

            `knickpointfinder` will identify more than one knickpoint
            per iteration, so this argument does not completely
            control the number of knickpoints identified. The default
            is the number of nodes in the stream network, and the
            iteration count is always limited to the number of nodes
            in the stream network.

        Returns
        -------
        np.ndarray
            A logical node attribute list that is True for stream
            network nodes that are identified as knickpoints

        Example
        -------
        >>> import topotoolbox
        >>> import matplotlib.pyplot as plt
        >>> dem = topotoolbox.load_dem('bigtujunga')
        >>> fd = topotoolbox.FlowObject(dem)
        >>> s = topotoolbox.StreamObject(fd)
        >>> s = s.klargestconncomps(1)
        >>> z = topotoolbox.imposemin(s, dem)
        >>> kp = s.knickpointfinder(z, tolerance=50.0)
        >>> d = s.upstream_distance()
        >>> fig,ax = plt.subplots()
        >>> s.plotdz(dem, ax=ax, color='k')
        >>> ax.scatter(d[kp], z[kp])

        """
        z = self.ezgetnal(dem, dtype=np.float32)
        z = imposemin(self, z)

        # The number of knickpoints and thus the number of iterations
        # is bounded by the number of nodes in the stream network.
        if iterations is None:
            iterations = self.stream.size
        iterations = min(iterations, self.stream.size)

        nv = self.stream.size

        if knickpoints is None:
            kp = np.zeros(nv, dtype=np.bool)
        else:
            kp = np.array(knickpoints, copy=True)

        for _ in np.arange(iterations):
            zs = self.lowerenv(z, kp)
            dz = z - zs
            imax = np.arange(nv)

            # These loops compute the maximum difference between the
            # observed elevation and the lower convex envelope over
            # reaches separated by knickpoints. The first loop
            # propagates the maximum value and index downstream,
            # skipping edges that terminate on a knickpoint.
            for (u, v) in zip(self.source, self.target):
                if dz[v] < dz[u] and not kp[v]:
                    dz[v] = dz[u]
                    imax[v] = imax[u]

            # The second loop propagates the maximum back upstream, so
            # each connected component is labeled with the maximum
            # difference and the node index at which that maximum
            # occurs.
            for (u, v) in zip(reversed(self.source), reversed(self.target)):
                if dz[u] < dz[v] and not kp[v]:
                    dz[u] = dz[v]
                    imax[u] = imax[v]

            # These loops could be implemented in libtopotoolbox for
            # speed. The maximum-over-reaches logic could be useful
            # for other applications.

            # The unique node indices are the potential knickpoints
            # for each reach. We select only those whose difference
            # exceeds the user-provided tolerance.
            potential_knicks = np.unique(imax)
            new_knicks = dz[potential_knicks] >= tolerance

            # If no new knickpoints have been found, we stop
            # searching.
            if np.count_nonzero(new_knicks) == 0:
                break

            kp[potential_knicks] = new_knicks

        return kp

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __len__(self):
        return len(self.stream)

    def __iter__(self):
        return iter(self.stream)

    def __getitem__(self, index):
        return self.stream[index]

    def __setitem__(self, index, value):
        try:
            value = np.float32(value)
        except (ValueError, TypeError):
            raise TypeError(
                f"{value} can't be converted to float32.") from None

        self.stream[index] = value

    def __array__(self):
        return self.stream

    def __str__(self):
        return str(self.stream)
