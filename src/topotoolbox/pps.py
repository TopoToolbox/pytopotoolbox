"""Tools for analyzing point patterns on stream networks
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
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

    def plotdz(self, z, distance=None,
               ax=None,
               dunit: str = 'm', doffset: float = 0,
               scalex=True, scaley=True,
               **kwargs):
        """Plot upstream distances of points in a PPS against a covariate

        Note that this will only plot the points of the PPS. To
        additionally plot the stream profiles, use `p.s.plotdz(z,
        distance)` separately.

        Parameters
        ----------
        z: GridObject, np.ndarray
          The node attribute list that will be plotted on the y axis

        distance: GridObject, np.ndarray
          The node attribute list that will be plotted on the x axis

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
            Additional keyword arguments are forwarded to `scatter.`

        Returns
        -------
        matplotlib.axes.Axes
            The axes into which the plot as been added

        Raises
        ------
        ValueError
            If `dunit` is not one of 'm' or 'km'.

        Example
        -------
        .. plot ::

           >>> import topotoolbox
           >>> import matplotlib.pyplot as plt
           >>> dem = topotoolbox.load_dem('bigtujunga')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> s = topotoolbox.StreamObject(fd, threshold=1000)
           >>> s = s.klargestconncomps(1)
           >>> kp = s.knickpointfinder(dem, tolerance=20.0)
           >>> p = topotoolbox.PPS.from_nal(s, kp)
           >>> fig, ax = plt.subplots()
           >>> _ = s.plotdz(dem, ax=ax)
           >>> _ = p.plotdz(dem, ax=ax, color='k', zorder=2)
           >>> plt.show()
        """
        if ax is None:
            ax = plt.gca()

        if distance is None:
            distance = self.s.upstream_distance()

        dist = self.s.ezgetnal(distance)

        if dunit == 'km':
            dist /= 1000
        elif dunit != 'm':
            raise ValueError("dunit must be one of 'm' or 'km'")

        dist += doffset

        z = self.s.ezgetnal(z)
        ax.scatter(dist[self.pp], z[self.pp], **kwargs)
        ax.autoscale_view(scalex=scalex, scaley=scaley)

        return ax
