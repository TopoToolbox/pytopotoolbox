"""This module contains the StreamObject class.
"""
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
                 threshold: int | float | GridObject | np.ndarray = 1) -> None:
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
        The threshold for flow accumulation. This can be an integer, float, 
        GridObject, or a NumPy array. All values above the threshold will 
        be considered. Default is 1.

    Raises
    ------
    ValueError
        If the `units` parameter is not 'pixels', 'mapunits', 'm2', or 'km2'.
    ValueError
        If the shape of the threshold does not match the flow object shape.
        """

        if units not in ['pixels', 'mapunits', 'm2', 'km2']:
            err = (f"Invalid unit '{units}' provided. Expected one of "
                   f"'pixels', 'mapunits', 'm2', 'km2'.")
            raise ValueError(err) from None

        if isinstance(threshold, int) or isinstance(threshold, float):
            threshold = np.full(flow.shape, threshold)
        elif isinstance(threshold, np.ndarray):
            if threshold.shape != flow.shape:
                err = (f"Threshold array shape {threshold.shape} does not "
                       f"match flow shape {flow.shape}.")
                raise ValueError(err) from None
            threshold = threshold.astype(np.float32, order='F')
        else:
            if threshold.shape != flow.shape:
                err = (f"Threshold GridObject shape {threshold.shape} does "
                       f"not match flow shape {flow.shape}.")
                raise ValueError(err) from None

            threshold = threshold.z.astype(np.float32, order='F')

        # Generate the flow accumulation matrix (acc)
        acc = np.zeros_like(flow.target, order='F', dtype=np.float32)
        weights = np.ones_like(flow.target, order='F', dtype=np.float32)
        _flow.flow_accumulation(
            acc, flow.source, flow.direction, weights, flow.shape)

        # Generate a 1D array that holds all indexes where more water than
        # in the required threshold is collected. (acc >= threshold)
        threshold = threshold.flatten(order='F')
        acc = acc.flatten(order='F')
        temp = []
        for i, value in enumerate(acc):
            if value >= threshold[i]:
                temp.append(i)

        self.stream = np.array(temp, dtype=np.int32)

        # Based on the stream array, generate 3 1D arrays where the value of
        # the stream array at each index holds respective value of the
        # original array. (source, target and direction)
        source = flow.source.flatten(order='F')
        temp_source = []
        target = flow.target.flatten(order='F')
        temp_target = []
        direction = flow.direction.flatten(order='F')
        temp_direction = []

        for i in self.stream:
            temp_source.append(source[i])
            temp_target.append(target[i])
            temp_direction.append(direction[i])

        self.target = np.array(temp_target, dtype=np.int32)
        self.source = np.array(temp_source, dtype=np.int32)
        self.direction = np.array(temp_direction, dtype=np.int32)
        self.shape = flow.shape

        # misc
        self.path = flow.path
        self.name = flow.name

        # georeference
        self.bounds = flow.bounds
        self.transform = flow.transform
        self.crs = flow.crs

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
        stream = np.zeros(shape=self.shape, dtype=np.int32, order='F')
        for i in self.stream:
            x = i % self.shape[0]
            y = i // self.shape[0]
            stream[x][y] = 1

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
