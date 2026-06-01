"""Tools for analyzing point patterns on stream networks
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import scipy.stats as st

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

    def get_covariate(self, covariate=None):
        """Extract a covariate node attribute list

        Parameters
        ----------
        covariate: array_like, optional
          The covariate to be extracted. If it is not supplied, the
          upstream distance is returned. Otherwise, it should be
          either a node attribute list, a compatible GridObject or a
          compatible 2D np.ndarray, which will be selected at the
          points of the point process.

        """
        if covariate is None:
            return self.s.upstream_distance()

        return self.s.ezgetnal(covariate)

    def ksdensity(self, covariate=None, n_bootstraps=0, rng=np.random.default_rng()):
        """Estimate the distribution of covariate at the points using
        kernel density estimation.

        Parameters
        ----------
        covariate: array_like, optional
          The covariate of interest. If it is not supplied, the
          upstream distance is used. Otherwise, it should be
          either a node attribute list, a compatible GridObject or a
          compatible 2D np.ndarray, which will be selected at the
          points of the point process.

        n_bootstraps: int, optional
          The number of bootstrap samples used to compute confidence
          intervals. If not provided, no bootstrap samples are
          returned.

        rng: np.random.Generator
          A random number generator used to select bootstrap
          samples. Numpy's default_rng() is used by default.

        Returns
        -------
        x: np.ndarray
           The grid of points at which the densities are
           evaluated. The grid is based on the range of the covariate.

        n: np.ndarray
           The density estimate restricted to the points of the PPS

        nb: np.ndarray
           The density estimate throughout the stream network.

        ns: np.ndarray
           An array of size (npoints, n_bootstraps) containing the
           bootstrap sampled density estimates.

        bw: float
           The covariance factor used in the kernel density estimate routine.

        """
        c = self.get_covariate(covariate)

        n_ks = st.gaussian_kde(c[self.pp])
        bw = n_ks.covariance_factor()
        nb_ks= st.gaussian_kde(c, bw_method=bw)

        x = np.linspace(np.min(c), np.max(c), 1000)

        n = n_ks.pdf(x)
        nb = nb_ks.pdf(x)

        csim = rng.choice(c[self.pp], (self.npoints, n_bootstraps))
        ns = np.zeros((1000, n_bootstraps))
        for i in range(n_bootstraps):
            ns_ks = st.gaussian_kde(csim[:,i], bw_method=bw)
            ns[:,i] = ns_ks.pdf(x)

        return (x, n, nb, ns, bw)

    def rhohat(self, covariate=None, alpha=0.05, n_bootstraps=1000, rng=np.random.default_rng()):
        """Nonparametric estimation of point pattern dependence on covariate

        Parameters
        ----------
        covariate: array_like, optional
          The covariate of interest. If it is not supplied, the
          upstream distance is used. Otherwise, it should be
          either a node attribute list, a compatible GridObject or a
          compatible 2D np.ndarray, which will be selected at the
          points of the point process.

        alpha: float, optional
          The confidence level of the returned bootstrap confidence
          intervals. Default is 0.05

        n_bootstraps: int, optional
          The number of bootstrap samples. Default is 1000.

        rng: np.random.Generator
          A random number generator used to select bootstrap
          samples. Numpy's default_rng() is used by default.


        Returns
        -------
        x: np.ndarray
           The grid of points at which the densities are
           evaluated. The grid is based on the range of the covariate.

        rho: np.ndarray
           The estimated intensity

        rhol: np.ndarray
           The lower limit of the bootstrapped confidence interval

        rhou: np.ndarray
           The upper limit of the bootstrapped confidence interval

        Example
        -------
        .. plot ::

           >>> import topotoolbox
           >>> import matplotlib.pyplot as plt
           >>> dem = topotoolbox.load_dem('bigtujunga')
           >>> fd = topotoolbox.FlowObject(dem)
           >>> s = topotoolbox.StreamObject(fd, threshold=1000)
           >>> s = s.klargestconncomps(1)
           >>> kp = s.knickpointfinder(dem, tolerance=30.0)
           >>> p = topotoolbox.PPS.from_nal(s, kp)
           >>> a = fd.flow_accumulation()
           >>> c = s.chitransform(a)
           >>> x, rho, rhol, rhou = p.rhohat(c)
           >>> fig, ax = plt.subplots(1, 1)
           >>> _ = ax.plot(x, rho)
           >>> _ = ax.fill_between(x, rhol, rhou, alpha=0.5)
           >>> plt.show()
        """
        c = self.get_covariate(covariate)

        x, n, nb, ns, _ = self.ksdensity(c, n_bootstraps=n_bootstraps, rng=rng)

        intensity = self.intensity()

        nq = np.quantile(ns, [alpha/2, 1-alpha/2], axis=1)
        nl = nq[0, :]
        nu = nq[1, :]

        rho = intensity * n / nb
        rhou = intensity * nu / nb
        rhol = intensity * nl / nb

        return (x, rho, rhol, rhou)
