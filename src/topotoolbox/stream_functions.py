"""Functions for working with flow networks

These functions often apply to both StreamObjects and FlowObjects.
"""
import copy

import numpy as np

# pylint: disable=no-name-in-module
from . import _stream  # type: ignore
from . import StreamObject, FlowObject, GridObject

def imposemin(s, dem, minimum_slope=0.0):
    """Minima imposition along a drainage network

    Parameters
    ----------
    s : StreamObject | FlowObject
        The drainage network used to determine flow directions

    dem : GridObject | np.ndarray
        The elevations to be carved. If s is a FlowObject, this should
        either be a GridObject or a 2D array of the appropriate
        shape. If s is a StreamObject, this should be a GridObject or
        a 2D array of the shape of the DEM from which the StreamObject
        was derived or a 1D array (a node attribute list) with as many
        entries as there are nodes in the stream network.

    minimum_slope : float, optional
        The minimum downward gradient (expressed as a positive
        nondimensional slope) to be imposed on the
        elevations. Defaults to zero. Set to a small positive number
        (e.g. 0.001) to impose a shallow downward slope on the
        resulting stream profile. Too high a minimum gradient will
        result in channels that lie well below the land surface.

    Returns
    -------
    GridObject | np.ndarray
        The elevations with the minimum downward gradient imposed. If
        `dem` is a GridObject, a GridObject is returned. Otherwise an
        array of the same shape as `dem` is returned.
    """
    result = copy.deepcopy(s.ezgetnal(dem))

    z = np.asarray(result, dtype=np.float32, copy=False)

    d = -s.distance() * minimum_slope
    _stream.traverse_down_f32_min_add(z, d, s.source, s.target)

    return result
