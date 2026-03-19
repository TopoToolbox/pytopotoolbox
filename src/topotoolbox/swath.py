"""Python interface for swath profile analysis.

This module provides tools for computing and plotting transverse and
longitudinal swath profiles from Digital Elevation Models (DEMs).

Public functions
----------------
prepare_track
    Ensure a polyline track is D8-contiguous by gap-filling with Bresenham.
compute_swath_distance_map
    Perpendicular-distance map from a polyline; optionally compute the
    medial axis (Voronoi centreline) of the swath.
transverse_swath
    Aggregate DEM statistics into cross-track distance bins.
longitudinal_swath
    Aggregate DEM statistics along the track using a frontier Dijkstra
    nearest-point assignment and a sliding along-track window.
longitudinal_swath_windowed
    Same aggregation using an oriented rectangle window around each track
    point (no pre-computed distance map required).
get_point_pixels
    Retrieve the pixel coordinates assigned to a single track point by
    the frontier Dijkstra.
get_windowed_point_samples
    Retrieve the pixel coordinates inside the oriented rectangle window
    for a single track point.
rasterize_path
    Rasterize a set of ordered reference points into a dense pixel path.
simplify_line
    Simplify a dense polyline using one of three vertex-reduction methods.
"""
import math
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import depth_first_order
from .grid_object import GridObject
from .utils import transform_coords
# pylint: disable=no-name-in-module
from . import _swaths  # type: ignore


@dataclass
class SwathCentreLine:
    """Container for the outputs of ``compute_swath_distance_map``.

    Always populated
    ----------------
    distance_map : GridObject
        Perpendicular distance from each pixel to the nearest track segment,
        in metres.  When ``compute_signed=True`` (default), the sign follows
        the 2-D cross-product convention: **positive = left of the directed
        track**, negative = right.  Pixels outside ``half_width`` are NaN.
    nearest_point : GridObject, optional
        Integer index (0-based) of the nearest track point for each pixel.
        Always set when a ``SwathCentreLine`` is returned by
        ``compute_swath_distance_map``; required as input to
        ``longitudinal_swath`` and ``get_point_pixels``.

    Populated only when ``return_centre_line=True``
    ------------------------------------------------
    dist_from_boundary : GridObject, optional
        Distance (metres) from each swath pixel inward to the nearest swath
        boundary, computed by a D8 Dijkstra from the outer-edge pixels.
    centre_line_x : np.ndarray, optional
        Row indices (``indices2D``), flat indices (``indices1D``), or X
        coordinates (``coordinates``) of the detected medial-axis pixels,
        ordered along the track.
    centre_line_y : np.ndarray, optional
        Column indices or Y coordinates of medial-axis pixels (only set
        for ``indices2D`` and ``coordinates`` modes).
    centre_width : np.ndarray, optional
        Estimated local swath half-width (metres) at each medial-axis pixel,
        derived from the Voronoi ridge distances.
    """
    distance_map: GridObject
    nearest_point: Optional[GridObject] = None
    dist_from_boundary: Optional[GridObject] = None
    centre_line_x: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None
    centre_line_y: Optional[np.ndarray] = None
    centre_width: Optional[np.ndarray] = None


class TransverseSwath:
    """Aggregated cross-track statistics from ``transverse_swath``.

    Each attribute is a 1-D array with one entry per distance bin, covering
    the range ``[-half_width, +half_width]`` in steps of ``bin_resolution``.
    Bins with no valid pixels contain NaN (or 0 for ``counts``).

    Parameters
    ----------
    distances : np.ndarray
        Centre distance (metres) of each bin from the track.
        Negative values are to the right; positive values to the left.
    means : np.ndarray
        Mean elevation (metres) per bin.
    stddevs : np.ndarray
        Standard deviation of elevation per bin.
    mins : np.ndarray
        Minimum elevation per bin.
    maxs : np.ndarray
        Maximum elevation per bin.
    counts : np.ndarray of int
        Number of valid pixels in each bin.
    medians : np.ndarray, optional
        Median elevation per bin.  Populated when ``percentiles`` is given.
    q1, q3 : np.ndarray, optional
        25th and 75th percentile elevations per bin.
    percentiles : dict, optional
        Mapping ``{p: array}`` for each requested percentile ``p``.
    custom : np.ndarray, optional
        Per-bin result of a user-supplied ``custom_stat_fn``.
    """
    def __init__(self, distances, means, stddevs, mins, maxs, counts,
                 medians=None, q1=None, q3=None, percentiles=None, custom=None):
        self.distances = distances
        self.means = means
        self.stddevs = stddevs
        self.mins = mins
        self.maxs = maxs
        self.counts = counts
        self.medians = medians
        self.q1 = q1
        self.q3 = q3
        self.percentiles = percentiles
        self.custom = custom

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        """Return statistics for one bin as a dictionary.

        Parameters
        ----------
        idx : int
            Bin index.

        Returns
        -------
        dict
            Keys: ``distance``, ``mean``, ``std``, ``min``, ``max``,
            ``count``, and optionally ``median``, ``q1``, ``q3``,
            ``percentiles``.
        """
        d = {
            'distance': self.distances[idx],
            'mean': self.means[idx],
            'std': self.stddevs[idx],
            'min': self.mins[idx],
            'max': self.maxs[idx],
            'count': self.counts[idx]
        }
        if self.medians is not None:
            d['median'] = self.medians[idx]
        if self.q1 is not None:
            d['q1'] = self.q1[idx]
        if self.q3 is not None:
            d['q3'] = self.q3[idx]
        if self.percentiles is not None:
            d['percentiles'] = {p: arr[idx] for p, arr in self.percentiles.items()}
        return d


class LongitudinalSwath:
    """Along-track statistics from ``longitudinal_swath`` or
    ``longitudinal_swath_windowed``.

    Each attribute is a 1-D array with one entry per output track point
    (after applying ``skip``).  Points with no valid pixels have NaN
    statistics and a count of 0.

    Parameters
    ----------
    means : np.ndarray
        Mean elevation (metres) for each output track point.
    stddevs : np.ndarray
        Standard deviation of elevation.
    mins : np.ndarray
        Minimum elevation.
    maxs : np.ndarray
        Maximum elevation.
    counts : np.ndarray of int
        Number of valid pixels aggregated for each point.
    medians : np.ndarray, optional
        Median elevation.  Populated when ``percentiles`` is given.
    q1, q3 : np.ndarray, optional
        25th and 75th percentile elevations.
    percentiles : dict, optional
        Mapping ``{p: array}`` for each requested percentile ``p``.
    along_track_distances : np.ndarray, optional
        Cumulative distance (metres) from the first track point to each
        output point.
    track_x, track_y : np.ndarray, optional
        Spatial coordinates of the output track points, in the same format
        as the ``input_mode`` used to build the profile.
    """
    def __init__(self, means, stddevs, mins, maxs, counts,
                 medians=None, q1=None, q3=None, percentiles=None,
                 along_track_distances=None, track_x=None, track_y=None):
        self.means = means
        self.stddevs = stddevs
        self.mins = mins
        self.maxs = maxs
        self.counts = counts
        self.medians = medians
        self.q1 = q1
        self.q3 = q3
        self.percentiles = percentiles
        self.along_track_distances = along_track_distances
        self.track_x = track_x
        self.track_y = track_y

    def __len__(self):
        return len(self.means)

    def __getitem__(self, idx):
        """Return statistics for one track point as a dictionary.

        Parameters
        ----------
        idx : int
            Output track-point index (after ``skip``).

        Returns
        -------
        dict
            Keys: ``mean``, ``std``, ``min``, ``max``, ``count``,
            ``distance``, and optionally ``x``, ``y``, ``median``,
            ``q1``, ``q3``, ``percentiles``.
        """
        d = {
            'mean': self.means[idx],
            'std': self.stddevs[idx],
            'min': self.mins[idx],
            'max': self.maxs[idx],
            'count': self.counts[idx],
            'distance': (self.along_track_distances[idx]
                         if self.along_track_distances is not None else idx),
        }
        if self.track_x is not None:
            d['x'] = self.track_x[idx]
        if self.track_y is not None:
            d['y'] = self.track_y[idx]
        if self.medians is not None:
            d['median'] = self.medians[idx]
        if self.q1 is not None:
            d['q1'] = self.q1[idx]
        if self.q3 is not None:
            d['q3'] = self.q3[idx]
        if self.percentiles is not None:
            d['percentiles'] = {p: arr[idx] for p, arr in self.percentiles.items()}
        return d


def _order_d8_path(rows, cols):
    """Return an index array that sorts D8-connected pixels into path order.

    The goal in this swath context is to order newly generated centerline in
    an contiguous line.

    Builds a sparse D8 adjacency graph over the pixel set, finds an endpoint
    (a pixel with exactly one D8 neighbour), then traverses the path via DFS.
    Falls back to index 0 if no endpoint exists (closed loop).
    """
    n = len(rows)
    if n <= 1:
        return np.arange(n, dtype=np.intp)

    rows = np.asarray(rows, dtype=np.intp)
    cols = np.asarray(cols, dtype=np.intp)

    # Build a lookup grid: (row, col) → index in the array.
    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())
    nr = r_max - r_min + 3
    nc = c_max - c_min + 3
    lookup = np.full((nr, nc), -1, dtype=np.intp)
    lookup[rows - r_min + 1, cols - c_min + 1] = np.arange(n)

    # D8 offsets — 8 vectorised passes instead of a loop over n points.
    d8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    src_parts, dst_parts = [], []
    for di, dj in d8:
        nb = lookup[rows - r_min + 1 + di, cols - c_min + 1 + dj]
        mask = nb >= 0
        src_parts.append(np.where(mask)[0])
        dst_parts.append(nb[mask])

    src = np.concatenate(src_parts)
    dst = np.concatenate(dst_parts)
    graph = csr_matrix((np.ones(len(src), dtype=np.float32), (src, dst)),
                       shape=(n, n))

    # Endpoint = pixel with exactly one D8 neighbour.
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    endpoints = np.where(degrees == 1)[0]
    start = int(endpoints[0]) if len(endpoints) > 0 else 0

    return depth_first_order(graph, start, directed=False,
                             return_predecessors=False)


def _swap_if_c_order(grid, a, b):
    """For C-contiguous grids swap (a, b) → (b, a); for F-contiguous return as-is.

    Used to convert between (row, col) and the C library's (fast, slow) convention:
    - F-contiguous: fast=row, slow=col  → no swap needed
    - C-contiguous: fast=col, slow=row  → swap needed
    """
    if grid.z.flags.c_contiguous:
        return b, a
    return a, b

def _unwrap_z(x):
    """Return x.z if x is a GridObject, else x as-is."""
    return x.z if isinstance(x, GridObject) else x


def _cum_dist(ti, tj, cellsize):
    """Cumulative along-track distance array (float32, starts at 0)."""
    steps = np.sqrt(np.diff(ti)**2 + np.diff(tj)**2) * cellsize
    return np.concatenate(([0.0], np.cumsum(steps))).astype(np.float32)


def _pca_tangent(ti, tj, pt, half_n):
    """Dominant PCA direction at track point ``pt`` using ±half_n neighbours."""
    n = len(ti)
    lo = max(0, pt - half_n)
    hi = min(n - 1, pt + half_n)
    if hi - lo < 1:
        lo, hi = (pt, pt + 1) if pt < n - 1 else (pt - 1, pt)
    seg_i, seg_j = ti[lo:hi + 1], tj[lo:hi + 1]
    mi, mj = seg_i.mean(), seg_j.mean()
    di, dj = seg_i - mi, seg_j - mj
    cii = float(np.dot(di, di))
    cij = float(np.dot(di, dj))
    cjj = float(np.dot(dj, dj))
    diff = cii - cjj
    D = np.sqrt(diff * diff + 4.0 * cij * cij)
    vi, vj = (diff + D, 2.0 * cij) if cii >= cjj else (2.0 * cij, -diff + D)
    vlen = np.sqrt(vi * vi + vj * vj)
    if vlen > 1e-10:
        return vi / vlen, vj / vlen
    d_i, d_j = ti[hi] - ti[lo], tj[hi] - tj[lo]
    length = np.sqrt(d_i * d_i + d_j * d_j)
    return (d_i / length, d_j / length) if length > 0 else (1.0, 0.0)


def _prepare_track(grid, track_x, track_y, input_mode):
    """Convert track inputs to float32 (row, col) pixel index arrays.

    Supports three input conventions controlled by ``input_mode``:
    ``"indices2D"`` (row/col arrays), ``"indices1D"`` (flat linear indices),
    and ``"coordinates"`` (georeferenced X/Y via the grid transform).
    """
    if input_mode == "indices2D":
        ti = np.array(track_x, dtype=np.float32)
        tj = np.array(track_y, dtype=np.float32)
    elif input_mode == "indices1D":
        idx = np.asarray(track_x)
        if grid.z.flags.f_contiguous:
            # F-order: idx = row + col * nrows
            ti = (idx % grid.rows).astype(np.float32)
            tj = (idx // grid.rows).astype(np.float32)
        else:
            # C-order: idx = row * ncols + col
            ti = (idx // grid.z.shape[1]).astype(np.float32)
            tj = (idx % grid.z.shape[1]).astype(np.float32)
    elif input_mode == "coordinates":
        inv_transform = ~grid.transform
        cols, rows = inv_transform * (np.asarray(track_x), np.asarray(track_y))
        tj = np.array(cols, dtype=np.float32)
        ti = np.array(rows, dtype=np.float32)
    else:
        raise ValueError(f"Invalid input_mode: {input_mode}")
    return ti, tj


def _grid_from_z(base_grid: GridObject, z: np.ndarray, name: str) -> GridObject:
    """Helper to create a GridObject from a Z array using metadata from a base grid."""
    out = GridObject()
    out.z = z
    out.cellsize = base_grid.cellsize
    out.bounds = base_grid.bounds
    out.transform = base_grid.transform
    out.georef = base_grid.georef
    out.name = name
    return out


def prepare_track(grid: GridObject, track_x, track_y=None, input_mode="indices2D"):
    """Ensure a track is D8-contiguous by gap-filling with Bresenham interpolation.

    Rounds each reference point to the nearest integer pixel, then checks
    consecutive pairs for D8 adjacency (``max(|Δrow|, |Δcol|) ≤ 1``).  If any
    gap is larger, the entire track is re-rasterized via the Bresenham D8
    algorithm so that every consecutive pair of output pixels is D8-adjacent.
    This is a prerequisite for the frontier Dijkstra used internally by
    ``compute_swath_distance_map``.

    Parameters
    ----------
    grid : GridObject
        Reference grid (used only for its transform and shape).
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates, depending on
        ``input_mode``.
    track_y : array-like, optional
        Column indices or Y coordinates.  Not used for ``"indices1D"``.
    input_mode : str, default ``"indices2D"``
        Coordinate convention: ``"indices2D"``, ``"indices1D"``, or
        ``"coordinates"``.

    Returns
    -------
    track_x : np.ndarray
        Gap-filled track in the same format as the input.
    track_y : np.ndarray or None
        Second coordinate array (``None`` for ``"indices1D"``).
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    ti_int = np.round(ti).astype(np.intp)
    tj_int = np.round(tj).astype(np.intp)

    di = np.abs(np.diff(ti_int))
    dj = np.abs(np.diff(tj_int))
    if np.all(np.maximum(di, dj) <= 1):
        oi = ti_int.astype(np.float32)
        oj = tj_int.astype(np.float32)
    else:
        max_size = int(np.sum(np.maximum(di, dj)) + len(ti_int))
        out_i = np.zeros(max_size, dtype=np.intp)
        out_j = np.zeros(max_size, dtype=np.intp)
        ci, cj = _swap_if_c_order(grid, ti_int, tj_int)
        count = _swaths.rasterize_path(out_i, out_j, ci, cj, 0, 0)
        raw_i, raw_j = _swap_if_c_order(grid, out_i[:count], out_j[:count])
        oi = raw_i.astype(np.float32)
        oj = raw_j.astype(np.float32)

    return transform_coords(grid, oi, oj, input_mode="indices2D", output_mode=input_mode, center=False)


def compute_swath_distance_map(
        grid: GridObject, track_x, track_y=None, half_width: Optional[float] = None,
        input_mode="indices2D", compute_signed=True,
        return_nearest_point=False, return_centre_line=False,
        mask=None) -> Union[GridObject, SwathCentreLine]:
    """Compute a perpendicular-distance map from a polyline track.

    For each grid pixel the function returns the shortest perpendicular
    distance to the nearest track segment, computed by a frontier Dijkstra
    that propagates outward from the rasterized track.

    Parameters
    ----------
    grid : GridObject
        Reference DEM; its shape, cellsize, and transform are used.
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates (see ``input_mode``).
    track_y : array-like, optional
        Column indices or Y coordinates.
    half_width : float, optional
        Swath half-width in metres.  Pixels farther than this are set to NaN.
        If ``None``, the full unclipped distance map is returned (pixels
        unreachable by the Dijkstra are NaN).
    input_mode : str, default ``"indices2D"``
        Coordinate convention: ``"indices2D"``, ``"indices1D"``, or
        ``"coordinates"``.
    compute_signed : bool, default True
        If ``True``, the distance is signed: **positive = left of the directed
        track**, negative = right (2-D cross-product convention).
        If ``False``, the absolute distance is returned.
    return_nearest_point : bool, default False
        If ``True``, the result is a ``SwathCentreLine`` whose
        ``nearest_point`` attribute holds the per-pixel nearest track-point
        index.  Required input to ``longitudinal_swath``.
    return_centre_line : bool, default False
        If ``True``, also compute the Voronoi medial axis of the swath (the
        centreline equidistant from both swath boundaries).  Implies
        ``return_nearest_point``; requires ``half_width`` to be set.
    mask : np.ndarray of int8, optional
        Binary pixel mask (1 = active, 0 = excluded).  NaN pixels in
        ``grid.z`` are always excluded regardless of this mask.

    Returns
    -------
    GridObject
        Distance map only (when ``return_nearest_point=False`` and
        ``return_centre_line=False``).
    SwathCentreLine
        Distance map plus optional geometric outputs (otherwise).
        ``nearest_point`` is always set when a ``SwathCentreLine`` is returned.

    Notes
    -----
    The Voronoi centreline (``return_centre_line=True``) is found by running
    two boundary Dijkstra waves from the positive and negative outer edges of
    the swath and marking pixels where the two wave-fronts meet.  The result
    is then thinned to D8-connectivity.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    hw_px = half_width / grid.cellsize if half_width is not None else float('inf')
    need_signed = compute_signed or return_centre_line

    near_pt_z = np.full(grid.z.shape, -1, dtype=np.intp)
    best_abs = np.empty(grid.z.shape, dtype=np.float32)
    signed_dist_px = np.empty(grid.z.shape, dtype=np.float32) if need_signed else None

    # Combine the caller mask with a NaN mask derived from the DEM so that
    # no-data pixels are never claimed by the frontier Dijkstra.
    nan_mask = (~np.isnan(grid.z)).astype(np.int8)
    if mask is not None:
        combined_mask = (nan_mask & mask.astype(np.int8)).astype(np.int8)
    else:
        combined_mask = nan_mask

    ci, cj = _swap_if_c_order(grid, ti, tj)
    _swaths.swath_frontier_distance_map(
        best_abs, signed_dist_px, near_pt_z, ci, cj, grid.dims, hw_px,
        combined_mask
    )

    # Determine which pixels are "outside" the swath:
    #   clipped (half_width given): pixels beyond the radius, including those
    #     that hit FLT_MAX because the Dijkstra cost exceeded hw_px.
    #   full (half_width=None): only truly unvisited pixels (FLT_MAX); hw_px=inf
    #     means the Dijkstra never discards a pixel based on distance alone.
    if half_width is not None:
        outside = best_abs > hw_px
    else:
        outside = best_abs >= np.finfo(np.float32).max

    # Convert pixel-unit distances to metres and mask outside pixels as NaN.
    dist_src = signed_dist_px if (compute_signed and signed_dist_px is not None) else best_abs
    dist_z = np.where(outside, np.nan, dist_src * grid.cellsize).astype(np.float32)
    dist_grid = _grid_from_z(grid, dist_z, "swath_distance")

    near_pt_grid = _grid_from_z(grid, near_pt_z, "swath_nearest_point")

    if not return_centre_line:
        if not return_nearest_point:
            return dist_grid
        return SwathCentreLine(distance_map=dist_grid, nearest_point=near_pt_grid)

    # --- Voronoi centreline ---
    # 1. Find the outer boundary of the swath mask via 4-connectivity padding.
    #    Boundary pixels are inside-mask pixels that have at least one
    #    outside-mask (or out-of-bounds) 4-neighbour.
    inside = ~outside
    pad = np.pad(inside, 1, constant_values=False)
    boundary = inside & (~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:])
    # 2. Split the outer boundary by the sign of signed_dist, giving seeds for
    #    the two competing Dijkstra waves (positive half = left edge,
    #    negative half = right edge).
    pos_seeds = np.flatnonzero((boundary & (signed_dist_px >= 0)).ravel()).astype(np.intp)
    neg_seeds = np.flatnonzero((boundary & (signed_dist_px <= 0)).ravel()).astype(np.intp)

    # 3. Run boundary Dijkstra from each outer edge inward.
    swath_mask = inside.astype(np.int8)
    dist_pos = np.empty(grid.z.shape, dtype=np.float32)
    dist_neg = np.empty(grid.z.shape, dtype=np.float32)
    _swaths.swath_boundary_dijkstra(dist_pos, swath_mask, pos_seeds, grid.dims)
    _swaths.swath_boundary_dijkstra(dist_neg, swath_mask, neg_seeds, grid.dims)

    # 4. Inward distance from the boundary = min of both wave-front distances.
    mn = np.minimum(dist_pos, dist_neg)
    dfb_z = np.where(mn >= np.finfo(np.float32).max, np.nan, mn * grid.cellsize).astype(np.float32)

    cli_arr = np.empty(grid.z.size, dtype=np.float32)
    clj_arr = np.empty(grid.z.size, dtype=np.float32)
    cw_arr = np.empty(grid.z.size, dtype=np.float32)
    count = _swaths.voronoi_ridge_to_centreline(
        cli_arr, clj_arr, cw_arr, dist_pos, dist_neg, best_abs, hw_px,
        near_pt_z, ci, cj, grid.dims, grid.cellsize
    )
    count = _swaths.thin_rasterised_line_to_D8(cli_arr, clj_arr, cw_arr, count, grid.dims)

    res = SwathCentreLine(distance_map=dist_grid, nearest_point=near_pt_grid)
    res.dist_from_boundary = _grid_from_z(grid, dfb_z, "swath_dist_from_boundary")
    raw_i, raw_j = _swap_if_c_order(grid, cli_arr[:count], clj_arr[:count])
    order = _order_d8_path(raw_i.astype(np.intp), raw_j.astype(np.intp))
    oi, oj = raw_i[order], raw_j[order]
    out = transform_coords(grid, oi, oj, input_mode="indices2D", output_mode=input_mode, center=False)
    if input_mode == "indices1D":
        res.centre_line_x = out
    else:
        res.centre_line_x, res.centre_line_y = out
    res.centre_width = cw_arr[:count][order]
    return res


def transverse_swath(grid: GridObject, distance_map: Union[GridObject, np.ndarray],
                    half_width: float, bin_resolution: float = 10.0,
                    normalize: bool = False,
                    percentiles: Optional[List[int]] = None,
                    custom_stat_fn=None) -> TransverseSwath:
    """Compute a transverse (cross-track) swath profile.

    Pixels are assigned to distance bins based on their signed perpendicular
    distance to the track.  Each bin aggregates the elevations of all pixels
    whose distance falls within ``[bin_centre - bin_resolution/2,
    bin_centre + bin_resolution/2)``.

    Parameters
    ----------
    grid : GridObject
        Elevation DEM.  
    distance_map : GridObject or np.ndarray
        Signed distance map in **metres** from ``compute_swath_distance_map``
        (``compute_signed=True``).  Positive values are to the left of the
        directed track, negative to the right.
    half_width : float
        Swath half-width in metres.  Pixels outside ``[-half_width,
        +half_width]`` are excluded.
    bin_resolution : float, default 10.0
        Width of each distance bin in metres.
    normalize : bool, default False
        If ``True``, subtract the mean elevation of pixels within one bin of
        the track centre before aggregating, so the profile is relative.
    percentiles : list of int, optional
        Percentiles in the range 0–100 to compute for each bin.  When given,
        ``TransverseSwath.medians``, ``q1``, and ``q3`` are also populated.
    custom_stat_fn : callable, optional
        ``fn(values: np.ndarray) -> scalar`` called on the raw elevation
        values in each non-empty bin.  Result stored in
        ``TransverseSwath.custom``; NaN for empty bins.

    Returns
    -------
    TransverseSwath
        Per-bin statistics over the cross-track direction.
    """
    dist_arr = _unwrap_z(distance_map).ravel()
    dem = grid.z.ravel()

    if np.nanmin(dist_arr) >= 0:
        warnings.warn("Distance map appears to be absolute. Results will only cover positive side.")

    n_bins = int(2.0 * half_width / bin_resolution) + 1

    # Valid pixel mask
    valid = ~np.isnan(dem) & ~np.isnan(dist_arr) & (np.abs(dist_arr) <= half_width)
    dem_v = dem[valid].astype(np.float64)
    dist_v = dist_arr[valid].astype(np.float64)

    # Normalization: reference elevation = mean of pixels within one bin of track centre
    ref_elev = 0.0
    if normalize:
        ref_mask = valid & (np.abs(dist_arr) <= bin_resolution)
        if np.any(ref_mask):
            ref_elev = float(np.nanmean(dem[ref_mask]))
        dem_v = dem_v - ref_elev

    # Bin indices — matches C formula: int((dist + half_width) / bin_resolution + 0.5)
    bins_v = np.floor((dist_v + half_width) / bin_resolution + 0.5).astype(int)
    bins_v = np.clip(bins_v, 0, n_bins - 1)

    # Per-bin mean, std, count via bincount (single pass each, O(n))
    bin_counts = np.bincount(bins_v, minlength=n_bins).astype(np.int64)
    bin_sum    = np.bincount(bins_v, weights=dem_v, minlength=n_bins)
    bin_sum2   = np.bincount(bins_v, weights=dem_v * dem_v, minlength=n_bins)

    has_data = bin_counts > 0
    safe_counts = np.where(has_data, bin_counts, 1)  # avoid divide-by-zero
    bin_means = np.where(has_data, bin_sum / safe_counts, np.nan).astype(np.float32)
    var = np.where(has_data, bin_sum2 / safe_counts - (bin_sum / safe_counts) ** 2, np.nan)
    bin_stds = np.where(has_data, np.sqrt(np.maximum(var, 0.0)), np.nan).astype(np.float32)

    # Per-bin min/max via unbuffered scatter (np.minimum.at handles duplicate indices correctly)
    tmp_min = np.full(n_bins, np.inf)
    tmp_max = np.full(n_bins, -np.inf)
    if len(bins_v) > 0:
        np.minimum.at(tmp_min, bins_v, dem_v)
        np.maximum.at(tmp_max, bins_v, dem_v)
    bin_min = np.where(has_data, tmp_min + ref_elev, np.nan).astype(np.float32)
    bin_max = np.where(has_data, tmp_max + ref_elev, np.nan).astype(np.float32)

    bin_distances = np.array([-half_width + b * bin_resolution for b in range(n_bins)],
                              dtype=np.float32)

    # Percentiles and custom_stat_fn both require per-bin grouped values.
    # Sort once by bin index, then slice contiguous ranges.
    bin_medians = bin_q1 = bin_q3 = None
    perc_dict = None
    custom = None

    need_groups = (percentiles is not None) or (custom_stat_fn is not None)
    if need_groups and len(bins_v) > 0:
        order = np.argsort(bins_v, kind='stable')
        sorted_bins = bins_v[order]
        sorted_vals = dem_v[order]
        unique_bins, starts = np.unique(sorted_bins, return_index=True)
        ends = np.append(starts[1:], len(sorted_bins))

        if percentiles is not None:
            bin_medians = np.full(n_bins, np.nan, dtype=np.float32)
            bin_q1      = np.full(n_bins, np.nan, dtype=np.float32)
            bin_q3      = np.full(n_bins, np.nan, dtype=np.float32)
            perc_dict   = {p: np.full(n_bins, np.nan, dtype=np.float32) for p in percentiles}
        if custom_stat_fn is not None:
            custom = np.full(n_bins, np.nan)

        for idx, b in enumerate(unique_bins):
            vals = sorted_vals[starts[idx]:ends[idx]]
            if percentiles is not None:
                bin_medians[b] = np.percentile(vals, 50) + ref_elev
                bin_q1[b]      = np.percentile(vals, 25) + ref_elev
                bin_q3[b]      = np.percentile(vals, 75) + ref_elev
                for p in percentiles:
                    perc_dict[p][b] = np.percentile(vals, p) + ref_elev
            if custom_stat_fn is not None:
                custom[b] = custom_stat_fn(vals)

    return TransverseSwath(bin_distances, bin_means, bin_stds, bin_min, bin_max,
                           bin_counts, bin_medians, bin_q1, bin_q3, perc_dict, custom)


def longitudinal_swath(grid: GridObject, track_x, track_y,
                      distance_map: Union[GridObject, np.ndarray],
                      half_width: float, binning_distance: float,
                      nearest_point,
                      percentiles: Optional[List[int]] = None,
                      input_mode: str = "indices2D", skip: int = 1) -> LongitudinalSwath:
    """Compute a longitudinal (along-track) swath profile.

    Each output track point aggregates all swath pixels whose nearest track
    point (from the frontier Dijkstra) falls within a sliding window of
    ``±binning_distance`` metres of that point along the track.  When
    ``binning_distance=0``, each output point only includes pixels assigned
    directly to it (no overlap with neighbours).

    Parameters
    ----------
    grid : GridObject
        Elevation DEM.
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates (see ``input_mode``).
    track_y : array-like
        Column indices or Y coordinates.
    distance_map : GridObject or np.ndarray
        Signed distance map in **metres** from ``compute_swath_distance_map``
        (``compute_signed=True``).  Used to exclude pixels beyond
        ``half_width``.
    half_width : float
        Cross-track half-width in metres.
    binning_distance : float
        Along-track window half-length in metres.  Set to 0 for
        no overlap between neighbouring track points.
    nearest_point : GridObject or np.ndarray
        Per-pixel nearest track-point index from ``compute_swath_distance_map``
        (the ``SwathCentreLine.nearest_point`` attribute).
    percentiles : list of int, optional
        Percentiles in the range 0–100 to compute per track point.
    input_mode : str, default ``"indices2D"``
        Coordinate convention for ``track_x``/``track_y``.
    skip : int, default 1
        Output every ``skip``-th track point; useful for long tracks.

    Returns
    -------
    LongitudinalSwath
        Per-track-point statistics along the swath.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    n_points = len(ti)
    n_out = math.ceil(n_points / skip)

    pt_means = np.zeros(n_out, dtype=np.float32)
    pt_std = np.zeros(n_out, dtype=np.float32)
    pt_min = np.zeros(n_out, dtype=np.float32)
    pt_max = np.zeros(n_out, dtype=np.float32)
    pt_counts = np.zeros(n_out, dtype=np.int64)
    pt_medians = np.zeros(n_out, dtype=np.float32)
    pt_q1 = np.zeros(n_out, dtype=np.float32)
    pt_q3 = np.zeros(n_out, dtype=np.float32)

    perc_list = None
    pt_percs = None
    if percentiles is not None:
        perc_list = np.array(percentiles, dtype=np.int32)
        pt_percs = np.zeros((n_out, len(percentiles)), dtype=np.float32)

    res_i = np.zeros(n_out, dtype=np.float32)
    res_j = np.zeros(n_out, dtype=np.float32)

    dist_arr = _unwrap_z(distance_map)
    npt_arr = _unwrap_z(nearest_point)

    full_along_track = _cum_dist(ti, tj, grid.cellsize)

    ci, cj = _swap_if_c_order(grid, ti, tj)
    written = _swaths.swath_longitudinal(
        pt_means, pt_std, pt_min, pt_max, pt_counts,
        pt_medians, pt_q1, pt_q3, perc_list, pt_percs,
        grid.z, ci, cj,
        dist_arr, grid.dims, grid.cellsize, half_width, binning_distance,
        npt_arr, full_along_track, int(skip), res_i, res_j
    )

    res_i, res_j = _swap_if_c_order(grid, res_i[:written], res_j[:written])
    along_track = full_along_track[::skip][:written]

    perc_dict = None
    if percentiles is not None:
        perc_dict = {p: pt_percs[:written, i] for i, p in enumerate(percentiles)}

    out = transform_coords(grid, res_i, res_j, input_mode="indices2D", output_mode=input_mode, center=False)
    if input_mode == "indices1D":
        track_x_out, track_y_out = out, None
    else:
        track_x_out, track_y_out = out

    return LongitudinalSwath(pt_means[:written], pt_std[:written], pt_min[:written],
                             pt_max[:written], pt_counts[:written], pt_medians[:written],
                             pt_q1[:written], pt_q3[:written], perc_dict, along_track,
                             track_x_out, track_y_out)


def longitudinal_swath_windowed(grid: GridObject, track_x, track_y,
                                      half_width: float, binning_distance: float,
                                      n_points_regression: int = 5,
                                      percentiles: Optional[List[int]] = None,
                                      input_mode: str = "indices2D",
                                      skip: int = 1) -> LongitudinalSwath:
    """Compute a longitudinal swath profile using an oriented bounding-box window.

    Unlike ``longitudinal_swath``, this function does **not** require a
    pre-computed distance map or nearest-point array.  For each track point
    it:

    1. Estimates a local tangent direction by PCA on the ``n_points_regression``
       nearest track points.
    2. Projects all pixels in a bounding box onto the tangent (along-track)
       and orthogonal (cross-track) axes.
    3. Accumulates statistics for pixels inside the oriented rectangle
       ``[-binning_distance, +binning_distance] × [-half_width, +half_width]``
       (in pixel units).

    This approach handles curved tracks correctly but is slower than
    ``longitudinal_swath`` for long tracks because the bounding-box search
    is repeated independently for each output point.

    Parameters
    ----------
    grid : GridObject
        Elevation DEM.
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates (see ``input_mode``).
    track_y : array-like
        Column indices or Y coordinates.
    half_width : float
        Cross-track half-width in metres.
    binning_distance : float
        Along-track window half-length in metres.
    n_points_regression : int, default 5
        Number of track points used for the local PCA tangent estimate.
        A larger value smooths the tangent direction at the cost of accuracy
        near sharp bends.
    percentiles : list of int, optional
        Percentiles in the range 0–100 to compute per track point.
    input_mode : str, default ``"indices2D"``
        Coordinate convention for ``track_x``/``track_y``.
    skip : int, default 1
        Output every ``skip``-th track point.

    Returns
    -------
    LongitudinalSwath
        Per-track-point statistics along the swath.

    Notes
    -----
    The PCA tangent is the principal eigenvector of the 2×2 covariance matrix
    of the local track segment coordinates.  When the local segment is
    degenerate (all points collinear or length zero) the function falls back
    to a simple finite-difference direction.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    dem = np.asarray(grid.z, dtype=np.float32)
    nrows, ncols = grid.z.shape
    cellsize = grid.cellsize
    hw_px = half_width / cellsize
    bd_px = binning_distance / cellsize

    n_pts = len(ti)
    half_n = max(1, n_points_regression // 2)
    tang_i = np.empty(n_pts, dtype=np.float32)
    tang_j = np.empty(n_pts, dtype=np.float32)
    for pt in range(n_pts):
        tang_i[pt], tang_j[pt] = _pca_tangent(ti, tj, pt, half_n)

    pts = range(0, n_pts, skip)
    n_out = len(pts)

    pt_means   = np.full(n_out, np.nan, dtype=np.float32)
    pt_std     = np.full(n_out, np.nan, dtype=np.float32)
    pt_min     = np.full(n_out, np.nan, dtype=np.float32)
    pt_max     = np.full(n_out, np.nan, dtype=np.float32)
    pt_counts  = np.zeros(n_out, dtype=np.int64)
    pt_medians = np.full(n_out, np.nan, dtype=np.float32)
    pt_q1      = np.full(n_out, np.nan, dtype=np.float32)
    pt_q3      = np.full(n_out, np.nan, dtype=np.float32)
    pt_percs   = None
    if percentiles is not None:
        pt_percs = np.full((n_out, len(percentiles)), np.nan, dtype=np.float32)

    # Row/col index grids (reused each iteration via slicing)
    R = hw_px + bd_px

    for out_idx, pt in enumerate(pts):
        ci, cj = ti[pt], tj[pt]
        t_i, t_j = tang_i[pt], tang_j[pt]
        o_i, o_j = -t_j, t_i  # orthogonal

        # Bounding box
        pi_lo = max(0,        int(ci - R))
        pi_hi = min(nrows - 1, int(ci + R) + 1)
        pj_lo = max(0,        int(cj - R))
        pj_hi = min(ncols - 1, int(cj + R) + 1)

        # Sub-array pixel coordinates
        pi_idx = np.arange(pi_lo, pi_hi + 1, dtype=np.float32)
        pj_idx = np.arange(pj_lo, pj_hi + 1, dtype=np.float32)
        pi_grid, pj_grid = np.meshgrid(pi_idx, pj_idx, indexing='ij')

        d_i = pi_grid - ci
        d_j = pj_grid - cj

        mask = (
            (np.abs(d_i * t_i + d_j * t_j) <= bd_px) &
            (np.abs(d_i * o_i + d_j * o_j) <= hw_px)
        )

        # meshgrid with indexing='ij' gives arrays shaped (n_rows, n_cols),
        # matching the DEM sub-array directly.
        vals = dem[pi_lo:pi_hi + 1, pj_lo:pj_hi + 1][mask]
        valid = ~np.isnan(vals)
        vals = vals[valid]

        count = len(vals)
        pt_counts[out_idx] = count
        if count == 0:
            continue

        pt_means[out_idx]  = vals.mean()
        pt_std[out_idx]    = vals.std()
        pt_min[out_idx]    = vals.min()
        pt_max[out_idx]    = vals.max()

        pt_medians[out_idx] = np.percentile(vals, 50)
        pt_q1[out_idx]      = np.percentile(vals, 25)
        pt_q3[out_idx]      = np.percentile(vals, 75)
        if percentiles is not None:
            for p_idx, p in enumerate(percentiles):
                pt_percs[out_idx, p_idx] = np.percentile(vals, p)

    along_track = _cum_dist(ti, tj, cellsize)[::skip][:n_out]

    perc_dict = None
    if percentiles is not None:
        perc_dict = {p: pt_percs[:, i] for i, p in enumerate(percentiles)}

    track_i_out = ti[::skip][:n_out]
    track_j_out = tj[::skip][:n_out]

    out = transform_coords(grid, track_i_out, track_j_out, input_mode="indices2D",
                           output_mode=input_mode, center=False)
    if input_mode == "indices1D":
        track_x_out, track_y_out = out, None
    else:
        track_x_out, track_y_out = out

    return LongitudinalSwath(pt_means, pt_std, pt_min, pt_max, pt_counts,
                             pt_medians, pt_q1, pt_q3, perc_dict, along_track,
                             track_x_out, track_y_out)


def get_point_pixels(grid: GridObject, track_x, track_y,
                    distance_map: Union[GridObject, np.ndarray],
                    point_index: int, half_width: float,
                    binning_distance: float,
                    nearest_point,
                    input_mode: str = "indices2D"):
    """Return the pixels assigned to a single track point by the frontier Dijkstra.

    Mirrors the pixel-selection logic of ``longitudinal_swath``: a pixel is
    returned if its nearest track point falls within ``±binning_distance``
    (metres) of ``point_index`` along the track, and its perpendicular distance
    is within ``half_width``.

    Parameters
    ----------
    grid : GridObject
        Reference grid; provides cellsize and shape.
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates (see ``input_mode``).
    track_y : array-like
        Column indices or Y coordinates.
    distance_map : GridObject or np.ndarray
        Signed distance map in metres from ``compute_swath_distance_map``.
    point_index : int
        0-based index of the track point to query.
    half_width : float
        Cross-track half-width in metres.
    binning_distance : float
        Along-track window half-length in metres (use 0 for the single-point
        assignment, i.e. pixels with ``nearest_point == point_index`` only).
    nearest_point : GridObject or np.ndarray
        Per-pixel nearest track-point index from ``compute_swath_distance_map``.
    input_mode : str, default ``"indices2D"``
        Coordinate convention for output.

    Returns
    -------
    tuple of np.ndarray
        ``(rows, cols)`` for ``"indices2D"``, flat indices for
        ``"indices1D"``, or ``(xs, ys)`` for ``"coordinates"``.
    """
    dist_arr = _unwrap_z(distance_map)
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    npt_arr = _unwrap_z(nearest_point)

    pi = np.zeros(grid.z.size, dtype=np.intp)
    pj = np.zeros(grid.z.size, dtype=np.intp)

    ci, cj = _swap_if_c_order(grid, ti, tj)
    count = _swaths.swath_get_point_pixels(
        pi, pj, ci, cj, point_index, dist_arr, grid.dims,
        grid.cellsize, half_width, binning_distance, npt_arr,
        _cum_dist(ti, tj, grid.cellsize)
    )

    oi, oj = _swap_if_c_order(grid, pi[:count], pj[:count])
    return transform_coords(grid, oi, oj, input_mode="indices2D", output_mode=input_mode, center=False)


def get_windowed_point_samples(grid: GridObject, track_x, track_y,
                               point_index: int, half_width: float,
                               binning_distance: float,
                               n_points_regression: int = 5,
                               input_mode: str = "indices2D"):
    """Return the pixels inside the oriented rectangle window for one track point.

    Mirrors the pixel-selection logic of ``longitudinal_swath_windowed``:
    estimates a local PCA tangent at ``point_index`` and returns all pixels
    within the axis-aligned bounding box that also satisfy
    ``|along_track| ≤ binning_distance`` and ``|cross_track| ≤ half_width``
    (in pixel units derived from the cellsize).

    Parameters
    ----------
    grid : GridObject
        Reference grid; provides cellsize, shape, and transform.
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates (see ``input_mode``).
    track_y : array-like
        Column indices or Y coordinates.
    point_index : int
        0-based index of the track point to query.
    half_width : float
        Cross-track half-width in metres.
    binning_distance : float
        Along-track window half-length in metres.
    n_points_regression : int, default 5
        Number of surrounding track points used for the local PCA tangent.
    input_mode : str, default ``"indices2D"``
        Coordinate convention for output.

    Returns
    -------
    tuple of np.ndarray
        ``(rows, cols)`` for ``"indices2D"``, flat indices for
        ``"indices1D"``, or ``(xs, ys)`` for ``"coordinates"``.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    n_pts = len(ti)
    hw_px = half_width / grid.cellsize
    bd_px = binning_distance / grid.cellsize
    nrows, ncols = grid.z.shape

    half_n = max(1, n_points_regression // 2)
    t_i, t_j = _pca_tangent(ti, tj, point_index, half_n)
    o_i, o_j = -t_j, t_i  # orthogonal

    ci, cj = ti[point_index], tj[point_index]
    R = hw_px + bd_px
    pi_lo = max(0,        int(ci - R))
    pi_hi = min(nrows - 1, int(ci + R) + 1)
    pj_lo = max(0,        int(cj - R))
    pj_hi = min(ncols - 1, int(cj + R) + 1)

    pi_idx = np.arange(pi_lo, pi_hi + 1, dtype=np.float32)
    pj_idx = np.arange(pj_lo, pj_hi + 1, dtype=np.float32)
    pi_grid, pj_grid = np.meshgrid(pi_idx, pj_idx, indexing='ij')
    d_i_grid = pi_grid - ci
    d_j_grid = pj_grid - cj
    mask = (
        (np.abs(d_i_grid * t_i + d_j_grid * t_j) <= bd_px) &
        (np.abs(d_i_grid * o_i + d_j_grid * o_j) <= hw_px)
    )
    oi = (pi_lo + np.where(mask)[0]).astype(np.intp)
    oj = (pj_lo + np.where(mask)[1]).astype(np.intp)
    return transform_coords(grid, oi, oj, input_mode="indices2D", output_mode=input_mode, center=False)


def rasterize_path(grid: GridObject, track_x, track_y=None,
                               input_mode="indices2D", close_loop=False,
                               use_d4=False):
    """Rasterize a path through ordered reference points using Bresenham's algorithm.

    Each consecutive pair of reference points is connected by a Bresenham line
    so that every output pixel is adjacent to the next in either D8 (default)
    or D4 connectivity.  Duplicate pixels at segment junctions are removed.

    Parameters
    ----------
    grid : GridObject
        Reference grid (used for its transform and shape).
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates of the reference
        points (see ``input_mode``).
    track_y : array-like, optional
        Column indices or Y coordinates.
    input_mode : str, default ``"indices2D"``
        Coordinate convention: ``"indices2D"``, ``"indices1D"``, or
        ``"coordinates"``.
    close_loop : bool, default False
        If ``True``, add a closing segment from the last point back to the
        first.
    use_d4 : bool, default False
        If ``True``, use D4 (4-connected) rasterization, which inserts extra
        cardinal steps at diagonal moves and roughly doubles the output size.

    Returns
    -------
    tuple of np.ndarray
        Rasterized path in the same format as the input.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)

    # Round to nearest pixel before integer conversion
    ti_int = np.round(ti).astype(np.intp)
    tj_int = np.round(tj).astype(np.intp)

    # Upper bound: sum of max(|di|,|dj|) for each segment pair, plus n_refs, times 2 for D4.
    di = np.abs(np.diff(ti_int))
    dj = np.abs(np.diff(tj_int))
    max_size = int(np.sum(np.maximum(di, dj)) + len(ti_int))
    if close_loop:
        max_size += int(max(np.abs(ti_int[-1] - ti_int[0]),
                            np.abs(tj_int[-1] - tj_int[0])) + 1)

    if use_d4:
        max_size *= 2

    out_i = np.zeros(max_size, dtype=np.intp)
    out_j = np.zeros(max_size, dtype=np.intp)

    ci, cj = _swap_if_c_order(grid, ti_int, tj_int)
    count = _swaths.rasterize_path(out_i, out_j, ci, cj,
                                               int(close_loop), int(use_d4))

    oi, oj = _swap_if_c_order(grid, out_i[:count], out_j[:count])
    return transform_coords(grid, oi, oj, input_mode="indices2D", output_mode=input_mode, center=False)


def simplify_line(grid: GridObject, track_x, track_y=None, tolerance: float = 1.0,
                  method: int = 0, input_mode: str = "indices2D"):
    """Simplify a polyline by reducing the number of vertices.

    Three methods are available, all preserving the first and last point.

    Parameters
    ----------
    grid : GridObject
        Reference grid (used for its transform and shape).
    track_x : array-like
        Row indices, flat 1-D indices, or X coordinates (see ``input_mode``).
    track_y : array-like, optional
        Column indices or Y coordinates.
    tolerance : float, default 1.0
        Interpretation depends on ``method``:

        * ``0`` (FIXED_N): target number of output points, clamped to
          ``[2, n_points]``.
        * ``1`` (KNEEDLE): ignored; the knee of the IEF error curve is
          detected automatically.
        * ``2`` (VW_AREA): minimum triangle area (pixel units²) to retain a
          vertex; larger values produce coarser simplification.
    method : int, default 0
        Simplification algorithm:
        ``0`` = FIXED_N (Iterative End-Point Fit, fixed target count),
        ``1`` = KNEEDLE (IEF with automatic knee detection),
        ``2`` = VW_AREA (Visvalingam–Whyatt area threshold).
    input_mode : str, default ``"indices2D"``
        Coordinate convention: ``"indices2D"``, ``"indices1D"``, or
        ``"coordinates"``.

    Returns
    -------
    tuple of np.ndarray
        Simplified track in the same format as the input.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    n_points = len(ti)

    out_i = np.zeros(n_points, dtype=np.float32)
    out_j = np.zeros(n_points, dtype=np.float32)

    ci, cj = _swap_if_c_order(grid, ti, tj)
    count = _swaths.simplify_line(out_i, out_j, ci, cj, tolerance, method)

    oi, oj = _swap_if_c_order(grid, out_i[:count], out_j[:count])
    return transform_coords(grid, oi, oj, input_mode="indices2D", output_mode=input_mode, center=False)
