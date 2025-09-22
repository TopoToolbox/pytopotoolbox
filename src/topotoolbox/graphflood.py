"""
Basic interface to libtopotoolbox implementation of graphflood.
"""

from copy import deepcopy

import numpy as np

# pylint: disable=no-name-in-module
from . import _graphflood  # type:ignore
from .grid_object import GridObject

# exposing function as string dictionary
funcdict = {
    "run_full": _graphflood.graphflood_run_full,
    "run_metrics": _graphflood.graphflood_metrics,
    "sfgraph": _graphflood.graphflood_sfgraph,
    "priority_flood_TO":
    _graphflood.compute_priority_flood_plus_topological_ordering,
    "priority_flood": _graphflood.compute_priority_flood,
    "drainage_area_single_flow": _graphflood.compute_drainage_area_single_flow,
}

__all__ = ["run_graphflood"]


def run_graphflood(
    grid: GridObject,
    initial_hw: np.ndarray | GridObject | None = None,
    bcs: np.ndarray | GridObject | None = None,
    dt: float = 1e-3,
    p: float | np.ndarray | GridObject = 10 * 1e-3 / 3600,
    manning: float | np.ndarray | GridObject = 0.033,
    sfd: bool = False,
    d8: bool = True,
    n_iterations: int = 100,
):
    """
    Runs the full graphflood's algorithm as described in Gailleton et al., 2024

    Parameters
    ----------
    grid : GridObject
        A GridObject representing the digital elevation model.
    initial_hw : np.ndarray or GridObject, optional
        Flow depth.
        Default is a matrix filled with zeros and the same shape as 'grid'.
    BCs : np.ndarray or GridObject, optional
        Boundary codes.
        Default is a matrix filled with ones except for the outermost edges,
        where values are set to 3, has same shape as 'grid'
    dt : float, optional
        time step(s ~ although this is not simulated time as we make the
        steady low assumption). Default is 1e-3.
    P : float, np.ndarray, or GridObject, optional
        Precipitation rates in m.s-1
        Default is a matrix with the same shape of 'grid'
        filled with 10 * 1e-3 / 3600.
    manning : float, np.ndarray, or GridObject, optional
        Friction coefficient.
        Default is a matrix with the same shape as 'grid' filled with 0.033.
    SFD : bool, optional
        True to compute single flow directions, False to compute multiple flow
        directions. Default is `False`.
    D8 : bool, optional
        True to include diagonal paths. Default is `True`.
    N_iterations : int, optional
        Number of iterations for the simulation. Default is 100.

    Returns
    -------
    GridObject
        A grid object with the computed water depths.

    Raises
    ------
    RuntimeError
        If the shape of `initial_hw`, `BCs`, `P`, or `manning` does not match
        the shape of the 'grid' GridObject`.

    Example
    -------
    >>> import topotoolbox
    >>> import matplotlib.pyplot as plt
    >>> import matplotlib.colors as colors
    >>> flood = topotoolbox.run_graphflood(topotoolbox.load_dem('perfectworld'))
    >>> im = flood.plot(norm=colors.SymLogNorm(linthresh=1e-3, linscale=1))
    >>> plt.show()
    """

    # Preparing the arguments
    ny = grid.rows
    nx = grid.columns
    dx = grid.cellsize
    dim = np.array([ny, nx], dtype=np.uint64)

    # Order C for vectorised topography
    z = grid.z.ravel(order="C")

    # Ingesting the flow depth
    if initial_hw is None:
        hw = np.zeros_like(z)
    else:
        if initial_hw.shape != grid.z.shape:
            raise RuntimeError(
                """Feeding the model with initial flow depth
                requires a 2D numpy array or a GridObject of
                the same dimension of the topographic grid"""
            )

        if isinstance(initial_hw, GridObject):
            hw = initial_hw.z.ravel(order="C")
        else:
            hw = initial_hw.ravel(order="C")

    # Ingesting boundary condition
    if bcs is None:
        tbcs = np.ones((grid.rows, grid.columns), dtype=np.uint8)
        tbcs[[0, -1], :] = 3
        tbcs[:, [0, -1]] = 3
        tbcs = tbcs.ravel(order="C")
    else:
        if bcs.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with boundary conditions requires
                a 2D numpy array or a GridObject of the same dimension
                 of the topographic grid"""
            )

        if isinstance(bcs, GridObject):
            tbcs = bcs.z.ravel(order="C").astype(np.uint8)
        else:
            tbcs = bcs.ravel(order="C").astype(np.uint8)

    # Ingesting Precipitations
    if isinstance(p, np.ndarray):
        if p.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with precipitations requires a
                2D numpy array or a GridObject of the same
                dimension of the topographic grid"""
            )
        precipitations = p.ravel(order="C")
    elif isinstance(p, GridObject):
        if p.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with precipitations requires a
                2D numpy array or a GridObject of the same dimension
                of the topographic grid"""
            )
        precipitations = p.z.ravel(order="C")
    else:
        # in case precipitation is a scalar
        precipitations = np.full_like(z, p)

    # Ingesting manning
    if isinstance(manning, np.ndarray):
        if manning.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with precipitations requires a
                2D numpy array or a GridObject of the same dimension
                 of the topographic grid"""
            )
        manning = manning.ravel(order="C")
    elif isinstance(manning, GridObject):
        if manning.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with precipitations requires a
                2D numpy array or a GridObject of the same dimension
                of the topographic grid"""
            )
        manning = manning.z.ravel(order="C")
    else:
        # in case precipitation is a scalar
        manning = np.full_like(z, manning)

    tz = z.astype(np.float64)
    thw = hw.astype(np.float64)
    precipitations = precipitations.astype(np.float64)
    manning = manning.astype(np.float64)

    _graphflood.graphflood_run_full(
        tz, thw, tbcs, precipitations, manning,
        dim, dt, dx, sfd, d8, n_iterations, 1e-3
    )


    qvol_i = np.zeros_like(thw)
    qvol_o = np.zeros_like(thw)
    qo = np.zeros_like(thw)
    u = np.zeros_like(thw)
    sw = np.zeros_like(thw)

    _graphflood.graphflood_metrics(
        tz, thw, tbcs, precipitations, manning,
        qvol_i, qvol_o, qo, u, sw, dim, dx, d8, 1e-3 )


    res = {}

    resw = deepcopy(grid)
    resw.z = thw.reshape(grid.shape) #.astype(np.float32)
    res['hw'] = deepcopy(resw)
    resw.z = qvol_i.reshape(grid.shape) #.astype(np.float32)
    res['Qi'] = deepcopy(resw)
    resw.z = qvol_o.reshape(grid.shape) #.astype(np.float32)
    res['Qo'] = deepcopy(resw)
    resw.z = qo.reshape(grid.shape) #.astype(np.float32)
    res['qo'] = deepcopy(resw)
    resw.z = u.reshape(grid.shape) #.astype(np.float32)
    res['u'] = deepcopy(resw)
    resw.z = sw.reshape(grid.shape) #.astype(np.float32)
    res['Sw'] = deepcopy(resw)

    return res
