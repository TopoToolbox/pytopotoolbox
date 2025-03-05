"""This module contains the StreamObject class.
"""
import math
import warnings
import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix

from .flow_object import FlowObject

# pylint: disable=no-name-in-module
from . import _flow  # type: ignore
from . import _stream  # type: ignore
from .grid_object import GridObject

_all_ = ['StreamObject']


class StreamObject():
    """A class to represent stream flow accumulation based on a FlowObject.
    """

    def __init__(self, flow: FlowObject, units: str = 'pixels',
                 threshold: int | float | GridObject | np.ndarray = 0,
                 stream_pixels: GridObject | np.ndarray | None = None) -> None:
        """
    Initializes the StreamObject by processing flow accumulation.

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

    Raises
    ------
    ValueError
        If the `units` parameter is not 'pixels', 'mapunits', 'm2', or 'km2'.
    ValueError
        If the shape of the threshold does not match the flow object shape.
        """
        if not isinstance(flow, FlowObject):
            err = f"{flow} is not a topotoolbox.FlowObject."
            raise TypeError(err)

        self.cellsize = flow.cellsize
        self.shape = flow.shape
        self.strides = flow.strides

        # georeference
        self.bounds = flow.bounds
        self.transform = flow.transform
        self.crs = flow.crs

        cell_area = 0.0
        # Calculate the are of a cell based on the units argument.
        if units == 'pixels':
            cell_area = 1.0
        elif units == 'm2':
            cell_area = self.cellsize**2
        elif units == 'km2':
            cell_area = (self.cellsize*0.001)**2
        elif units == 'mapunits':
            if self.crs is not None:
                if self.crs.is_projected:
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
            if stream_pixels.shape != self.shape:
                err = (
                    f"stream_pixels shape {stream_pixels.shape}"
                    f" does not match FlowObject shape {self.shape}.")
                raise ValueError(err)

            if isinstance(stream_pixels, GridObject):
                w = (stream_pixels.z != 0).ravel(order='F')

            elif isinstance(stream_pixels, np.ndarray):
                w = (stream_pixels != 0).ravel(order='F')

            if threshold != 0:
                warn = ("Since stream_pixels have been provided, the "
                        "input for threshold will be ignored.")
                warnings.warn(warn)

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
            elif isinstance(threshold, np.ndarray):
                if threshold.shape != self.shape:
                    err = (f"Threshold array shape {threshold.shape} does not "
                           f"match FlowObject shape: {self.shape}.")
                    raise ValueError(err) from None
                threshold = threshold.astype(np.float32, order='F')
            else:
                if threshold.shape != self.shape:
                    err = (
                        f"Threshold GridObject shape {threshold.shape} does "
                        f"not match FlowObject shape: {self.shape}.")
                    raise ValueError(err) from None

                threshold = threshold.z.astype(np.float32, order='F')

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

    def distance(self) -> np.ndarray:
        """
        Compute the pixel-to-pixel distance for each edge.

        Returns
        -------
        np.ndarray, float32
            An edge attribute list with the distance between pixels
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

    def ezgetnal(self,
                 k: GridObject | np.ndarray | float):
        """Retrieve a node attribute list from k

        Parameters
        ----------
        k : GridObject | np.ndarray | float
            The object from which node values will be extracted. If
            `k` is a `GridObject` or an `np.ndarray` with the same
            shape as the underlying DEM of this `StreamObject`, the
            node values will be extracted from the grid by
            indexing. If `k` is an array with the same shape as the
            node attribute list, `ezgetnal` returns `k`. If `k` is a
            scalar value, `ezgetnal` returns an array of the right
            shape filled with `k`.

        Raises
        ------
        ValueError
            If `k` does not have the right shape to be indexed by the
            `StreamObject`.
        TypeError
            If `k` does not represent a type of data that can be
            extracted into a node attribute list.
        """

        if isinstance(k, GridObject):
            nal = k.z[np.unravel_index(self.stream, self.shape, order='F')]
        elif isinstance(k, np.ndarray):
            if k.shape == self.shape:
                # We have passed an ndarray with the same shape as the
                # corresponding GridObject, index into it
                nal = k[np.unravel_index(self.stream, self.shape, order='F')]
            elif k.shape == self.stream.shape:
                # k is already a node attribute list
                nal = k
            else:
                raise ValueError(f"{k} is not of the appropriate shape")
        elif np.isscalar(k):
            nal = np.full(self.stream.shape, k)
        else:
            raise TypeError(
                f"{k} is not a supported source for a node attribute list")

        return nal

    def xy(self, data=None):
        """Compute the x and y coordinates of continuous stream segments

        Arguments
        ---------
        data: tuple, optional
           A tuple of two node attribute lists representing the
           desired x and y values for each pixel in the stream
           network. If this argument is not supplied, the returned x
           and y values are the indices of the pixel in the DEM in the
           second and first dimension respectively. This reversal of
           dimensions corresponds to the orientation used by pyplot's
           `imshow`, and allows plotting the stream network over a
           corresponding GridObject.

        Returns
        -------
        list
            A list of lists of (x,y) pairs.

        """
        if data is None:
            # pylint: disable=unbalanced-tuple-unpacking
            ys, xs = np.unravel_index(self.stream, self.shape, order='F')
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
        visited = np.zeros(self.stream.size, dtype=np.bool)
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

    def plot(self, ax=None, **kwargs):
        """Plot the StreamObject

        Stream segments as computed by StreamObject.xy are plotted
        using a LineCollection. Note that collections are not used in
        autoscaling the provided axis. If the axis limits are not
        already set, by another underlying plot, for example, call
        ax.autoscale_view() on the returned axes to show the plot.

        Parameters
        ----------
        ax: matplotlib.axes.Axes, optional
            The axes in which to plot the StreamObject. If no axes are
            given, the current axes are used.

        **kwargs
            Additional keyword arguments are forwarded to LineCollection

        Returns
        -------
        matplotlib.axes.Axes
            The axes into which the StreamObject has been plotted.
        """
        if ax is None:
            ax = plt.gca()
        collection = LineCollection(self.xy(), **kwargs)
        ax.add_collection(collection)
        return ax

    def plotdz(self, z, ax=None, dunit: str = 'm', doffset: float = 0, **kwargs):
        """Plot a node attribute list against upstream distance

        Note that collections are not used in
        autoscaling the provided axis. If the axis limits are not
        already set, by another underlying plot, for example, call
        ax.autoscale_view() on the returned axes to show the plot.

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
        dist = np.zeros_like(z)
        a = np.ones_like(z)

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

        result = copy.copy(self)
        result.stream = self.stream[trunks]
        cumsum_index = np.cumsum(trunks) - 1

        trunks = trunks[self.source] & trunks[self.target]

        result.source = self.source[trunks]
        result.target = self.target[trunks]

        result.source = cumsum_index[result.source]
        result.target = cumsum_index[result.target]

        return result

    def klargestconncomps(self, k = 1) -> 'StreamObject':
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
        """
        nv = self.stream.size
        ne = self.source.size

        # Compute outlets of the stream network
        # NOTE(wkearn): This can be factored into its own function (`streampoi`)
        indegree = np.zeros(nv, dtype=np.uint8)
        outdegree = np.zeros(nv, dtype=np.uint8)
        _stream.edgelist_degree(indegree, outdegree, self.source, self.target)
        outlets = (outdegree == 0) & (indegree > 0)

        # Count the nodes in each connected component of the stream
        # network.
        # This might be slightly inconsistent with the MATLAB
        # implementation which sorts by the number of edges in each
        # connected component.
        acc = np.ones(nv, dtype=np.float32)
        weights = np.ones(ne, dtype=np.float32)
        _stream.traverse_down_f32_add_mul(acc, weights, self.source, self.target)

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

        # NOTE(wkearn): this (copied from our `trunk` implementation)
        # is equivalent to `subgraph`, a STREAMobj method for
        # extracting a new StreamObject based on a logical node
        # attribute list.
        result = copy.copy(self)
        result.stream = self.stream[conncomps_mask]
        cumsum_index = np.cumsum(conncomps_mask) - 1

        conncomps_mask = conncomps_mask[self.source] & conncomps_mask[self.target]

        result.source = self.source[conncomps_mask]
        result.target = self.target[conncomps_mask]

        result.source = cumsum_index[result.source]
        result.target = cumsum_index[result.target]

        return result


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
