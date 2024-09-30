"""
Basic interface to libtopotoolbox implementation of graphflood

"""

from .grid_object import GridObject
import numpy as np
from ._graphflood import (  # type: ignore
    graphflood_run_full,
    graphflood_sfgraph,
    compute_priority_flood_plus_topological_ordering,
    compute_priority_flood,
    compute_drainage_area_single_flow
    )
from copy import deepcopy

# exposing function as string dictionary
funcdict = {
    "run_full": graphflood_run_full,
    "sfgraph": graphflood_sfgraph,
    "priority_flood_TO": compute_priority_flood_plus_topological_ordering,
    "priority_flood": compute_priority_flood,
    "drainage_area_single_flow": compute_drainage_area_single_flow,
}

__all__ = ["run_graphflood"]


def run_graphflood(
    grid: GridObject,
    initial_hw=None,
    BCs=None,
    dt=1e-3,
    P=10 * 1e-3 / 3600,
    manning=0.033,
    SFD=False,
    D8=True,
    N_iterations=100,
):

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

        if isinstance(grid, GridObject):
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

    graphflood_run_full(
        Z, hw, tBCs, Precipitations, manning,
        dim, dt, dx, SFD, D8, N_iterations, 1e-3
    )
    res = deepcopy(grid)

    res.z = hw.reshape(grid.shape)

    return res
