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
    BCs: np.ndarray | GridObject | None = None,
    dt: float = 1e-3,
    P: float | np.ndarray | GridObject = 10 * 1e-3 / 3600,
    manning: float | np.ndarray | GridObject = 0.033,
    SFD: bool = False,
    D8: bool = True,
    N_iterations: int = 100,
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
        [Add description]
        Default is `False`.
    D8 : bool, optional
        True to include diagonal paths. Default is `True`.
    N_iterations : int, optional
        Number of iterations for the simulation. Default is 100.

    Returns
    -------
    GridObject
        [Add description]

    Raises
    ------
    RuntimeError
        If the shape of `initial_hw`, `BCs`, `P`, or `manning` does not match
        the shape of the 'grid' GridObject`.
    """

    # Preparing the arguments
    ny = grid.rows
    nx = grid.columns
    dx = grid.cellsize
    dim = np.array([ny, nx], dtype=np.uint64)

    # Order C for vectorised topography
    Z = grid.z.ravel(order="C")

    # Ingesting the flow depth
    if initial_hw is None:
        hw = np.zeros_like(Z)
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
    if BCs is None:
        tBCs = np.ones((grid.rows, grid.columns), dtype=np.uint8)
        tBCs[[0, -1], :] = 3
        tBCs[:, [0, -1]] = 3
        tBCs = tBCs.ravel(order="C")
    else:
        if BCs.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with boundary conditions requires
                a 2D numpy array or a GridObject of the same dimension
                 of the topographic grid"""
            )

        if isinstance(BCs, GridObject):
            tBCs = BCs.z.ravel(order="C").astype(np.uint8)
        else:
            tBCs = BCs.ravel(order="C").astype(np.uint8)

    # Ingesting Precipitations
    if isinstance(P, np.ndarray):
        if P.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with precipitations requires a
                2D numpy array or a GridObject of the same
                dimension of the topographic grid"""
            )
        Precipitations = P.ravel(order="C")
    elif isinstance(P, GridObject):
        if P.shape != grid.shape:
            raise RuntimeError(
                """Feeding the model with precipitations requires a
                2D numpy array or a GridObject of the same dimension
                of the topographic grid"""
            )
        Precipitations = P.z.ravel(order="C")
    else:
        # in case precipitation is a scalar
        Precipitations = np.full_like(Z, P)

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
        manning = np.full_like(Z, manning)

    _graphflood.graphflood_run_full(
        Z, hw, tBCs, Precipitations, manning,
        dim, dt, dx, SFD, D8, N_iterations, 1e-3
    )
    res = deepcopy(grid)

    res.z = hw.reshape(grid.shape)

    return res
