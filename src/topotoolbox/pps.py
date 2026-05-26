"""Tools for analyzing point patterns on stream networks
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .stream_object import StreamObject


@dataclass
class PPS:
    """Point pattern on stream network

    An instance of PPS stores a stream network (StreamObject) and an
    associated point pattern.

    Implementation
    --------------

    Unlike the MATLAB implementation, a PPS stores a logical node
    attribute list indicating the presence or absence of a point on
    each node in the stream network.

    """

    s: StreamObject
    pp: npt.NDArray[np.intp]

    @property
    def npoints(self) -> int:
        """The number of points in the point pattern
        """
        return self.pp.size

    @property
    def tlength(self) -> float:
        """The total length of the stream network
        """
        return np.sum(self.s.node_to_node_distance())

    def intensity(self) -> float:
        """Estimate the intensity of points on the stream network

        The intensity is the expected number of points per unit length.
        """
        return self.npoints / self.tlength

    @classmethod
    def from_nal(cls, stream: StreamObject, nal: npt.NDArray[np.bool]):
        """Construct a PPS from a logical node attribute list
        """
        return cls(stream, np.flatnonzero(nal))
