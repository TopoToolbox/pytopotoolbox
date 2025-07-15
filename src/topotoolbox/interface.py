"""Generic TopoToolbox interface

This module contains functions that can be used with all TopoToolbox
objects.
"""

__all__ = ["validate_alignment"]


def validate_alignment(s1, s2) -> bool:
    """Check whether two TopoToolbox objects are aligned

    `validate_alignment` checks that the two objects have the same
    `shape` attribute and, if coordinate information is available, the
    same coordinate system given by the attributes `bounds`, `crs`
    and `transform`.

    Parameters
    ----------
    s1 : np.ndarray | GridObject | FlowObject | StreamObject
        The first object to check

    s2 : np.ndarray | GridObject | FlowObject | StreamObject
        The second object to check

    Returns
    -------
    bool
       True if the two objects are aligned, False otherwise

    Example
    -------
    >>> dem = topotoolbox.load_dem('bigtujunga')
    >>> fd = topotoolbox.FlowObject(dem)
    >>> print(topotoolbox.validate_alignment(dem, fd))
    """
    return (s1.shape == s2.shape) and all(
        (not hasattr(s1, attr) or not hasattr(s2, attr))
        or (getattr(s1, attr) == getattr(s2, attr))
        for attr in ["bounds", "georef", "transform"])
