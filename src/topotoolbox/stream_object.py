"""This module contains the StreamObject class.
"""
import math
import warnings

import numpy as np
import matplotlib.pyplot as plt

from .flow_object import FlowObject

# pylint: disable=no-name-in-module
from . import _flow  # type: ignore
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
        w = np.zeros(flow.shape,dtype='bool',order='F').ravel(order='K')
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
            acc = np.zeros_like(flow.target, order='F', dtype=np.float32)
            weights = np.ones_like(flow.target, order='F', dtype=np.float32)
            _flow.flow_accumulation(
                acc, flow.source, flow.direction, weights, flow.shape)

            # Generate a 1D array that holds all indexes where more water than
            # in the required threshold is collected. (acc >= threshold)
            w = (acc >= threshold).ravel(order='F')

        # Indices of pixels in the stream network
        # This is a node attribute list
        self.stream = np.nonzero(w)[0]

        # Find edges whose source pixel is in the stream network
        u = flow.source.ravel(order='F')
        v = flow.target.ravel(order='F')
        d = flow.direction.ravel(order='F')

        # v = -1 when the pixel is a sink or outlet. Drop those edges
        # from the flow network
        i = w[u] & (v != -1)

        # Renumber the nodes of the stream network
        ix = np.zeros_like(w,dtype='int64')
        ix[w] = np.arange(0,self.stream.size)

        # Edges in the stream network
        #
        # Elements of these edge attribute lists are 0-based indices
        # into node attribute lists.
        #
        # To convert these to pixel indices in the original GridObject
        # or FlowObject, use stream[source] or stream[target].
        self.source = ix[u[i]]
        self.target = ix[v[i]]
        self.direction = d[i]

        # misc
        self.path = flow.path
        self.name = flow.name

    def show(self, cmap='hot', overlay: GridObject | None = None,
             overlay_cmap: str = 'binary', alpha: float = 0.8) -> None:
        """
        Display the StreamObject instance as an image using Matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap that will be used for the stream.
        overlay_cmap : str, optional
            Matplotlib colormap that will be used in the background plot.
        overlay : GridObject | None, optional
            To overlay the stream over a dem to better visualize the stream.
        alpha : float, optional
            When using an dem to overlay, this controls the opacity of the dem.
        """
        stream = np.zeros(shape=self.shape, dtype=np.int64, order='F')
        stream[np.unravel_index(self.stream,self.shape,order='F')] = 1

        if overlay is not None:
            if self.shape == overlay.shape:
                plt.imshow(overlay, cmap=overlay_cmap, alpha=alpha)
                plt.imshow(stream, cmap=cmap,
                           alpha=stream.astype(np.float32))
                plt.show()
            else:
                err = (f"Shape mismatch: Stream shape {self.shape} does not "
                       f"match overlay shape {overlay.shape}.")
                raise ValueError(err) from None
        else:
            plt.imshow(stream, cmap=cmap)
            plt.title(self.name)
            plt.colorbar()
            plt.tight_layout()
            plt.show()

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
