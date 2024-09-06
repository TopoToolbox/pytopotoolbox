"""This module contains the StreamObject class.
"""
import numpy as np
import matplotlib.pyplot as plt

from .flow_object import FlowObject

# pylint: disable=no-name-in-module
from . import _flow  # type: ignore


_all_ = ['StreamObject']


class StreamObject():
    def __init__(self, flow: FlowObject) -> None:

        acc = np.zeros_like(flow.z, order='F', dtype=np.float32)
        weights = np.ones_like(flow.z, order='F', dtype=np.float32)

        _flow.flow_accumulation(
            acc, flow.source, flow.direction, weights, flow.shape)

        self.path = flow.path
        self.name = flow.name

        # raster metadata
        self.acc = acc
        self.shape = flow.shape

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
        plt.imshow(self.acc, cmap=cmap)
        plt.title(self.name)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    # 'Magic' functions:
    # ------------------------------------------------------------------------

    def __len__(self):
        return len(self.acc)

    def __iter__(self):
        return iter(self.acc)

    def __getitem__(self, index):
        return self.acc[index]

    def __setitem__(self, index, value):
        try:
            value = np.float32(value)
        except (ValueError, TypeError):
            raise TypeError(
                f"{value} can't be converted to float32.") from None

        self.acc[index] = value

    def __array__(self):
        return self.acc

    def __str__(self):
        return str(self.acc)
