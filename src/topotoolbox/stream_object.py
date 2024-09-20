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

        if units not in ['pixels', 'mapunits', 'm2', 'km2']:
            # TODO: ADD ERROR MESSAGE
            err = ""
            raise ValueError(err) from None

        if isinstance(threshold, int) or isinstance(threshold, float):
            threshold = np.full(flow.shape, threshold)
        elif isinstance(threshold, np.ndarray):
            if threshold.shape != flow.shape:
                # TODO: ADD ERROR MESSAGE
                err = ""
                raise ValueError(err) from None
            threshold = threshold(np.float32, order='F')
        else:
            if threshold.shape != flow.shape:
                # TODO: ADD ERROR MESSAGE
                err = ""
                raise ValueError(err) from None

            threshold = threshold.z(np.float32, order='F')

        acc = np.zeros_like(flow.target, order='F', dtype=np.float32)
        weights = np.ones_like(flow.target, order='F', dtype=np.float32)

        _flow.flow_accumulation(
            acc, flow.source, flow.direction, weights, flow.shape)

        temp = []
        threshold = threshold.flatten(order='F')
        acc = acc.flatten(order='F')
        for i, value in enumerate(acc):
            if value >= threshold[i]:
                temp.append(i)

        self.stream = np.array(temp, dtype=np.int32)

        # raster metadata
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

    def show(self, cmap='binary') -> None:
        """
        Display the StreamObject instance as an image using Matplotlib.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap that will be used in the plot.
        """
        stream = np.zeros(shape=self.shape, dtype=np.int32, order='F')
        for i in self.stream:
            x = i // self.shape[0]
            y = i % self.shape[0]
            stream[x][y] = 1

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
