"""
This module provides the Python interface for swath profile analysis.
It includes classes and functions for computing and plotting transverse
and longitudinal swath profiles from Digital Elevation Models (DEMs).
"""
import math
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
import numpy as np
from .grid_object import GridObject
# pylint: disable=no-name-in-module
from . import _swaths  # type: ignore


@dataclass
class SwathCentreLine:
    """Dataclass holding all outputs from a distance map with centre-line detection.

    Attributes
    ----------
    distance_map : GridObject
        Signed or absolute distance map from the track.
    nearest_point : GridObject, optional
        Map of nearest track point index for each pixel.
    dist_from_boundary : GridObject, optional
        Inward distance map from the swath boundaries.
    centre_line_x : np.ndarray, optional
        X-coordinates (or fast-dim indices) of the detected medial axis.
    centre_line_y : np.ndarray, optional
        Y-coordinates (or slow-dim indices) of the detected medial axis.
    centre_width : np.ndarray, optional
        Calculated local swath width at each centre-line point.
    """
    distance_map: GridObject
    nearest_point: Optional[GridObject] = None
    dist_from_boundary: Optional[GridObject] = None
    centre_line_x: Optional[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = None
    centre_line_y: Optional[np.ndarray] = None
    centre_width: Optional[np.ndarray] = None


class TransverseSwath:
    """Class representing a transverse swath profile (aggregated cross-section).

    Parameters
    ----------
    distances : np.ndarray
        Perpendicular distances of bin centers from the track.
    means : np.ndarray
        Mean elevation for each distance bin.
    stddevs : np.ndarray
        Standard deviation of elevation for each bin.
    mins : np.ndarray
        Minimum elevation for each bin.
    maxs : np.ndarray
        Maximum elevation for each bin.
    counts : np.ndarray
        Number of pixels sampled in each bin.
    medians : np.ndarray, optional
        Median elevation for each bin.
    q1, q3 : np.ndarray, optional
        25th and 75th percentiles for each bin.
    percentiles : dict, optional
        Dictionary mapping percentile values to arrays of results.
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
        self.percentiles = percentiles  # Dict mapping percentile value to array
        self.custom = custom            # Array of custom_stat_fn results per bin

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        """Returns a dictionary of statistics for a specific bin index."""
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

    def plot(self, fig=None, ax=None, show_minmax=True, show_std=False,
             show_quartiles=False, show_median=False, **kwargs):
        """Plot the transverse swath profile.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot into.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into.
        show_minmax : bool, default True
            Fill between min and max elevations.
        show_std : bool, default False
            Fill between mean ± standard deviation.
        show_quartiles : bool, default False
            Fill between Q1 and Q3.
        show_median : bool, default False
            Plot the median line.
        **kwargs
            Additional arguments passed to ax.plot.

        Returns
        -------
        fig, ax
            The figure and axes objects.
        """
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        label = kwargs.pop('label', 'Mean')
        line = ax.plot(self.distances, self.means, label=label, **kwargs)
        color = line[0].get_color()

        if show_minmax:
            ax.fill_between(self.distances, self.mins, self.maxs,
                            alpha=0.1, color=color, label='Min-Max')
        if show_std:
            ax.fill_between(self.distances, self.means - self.stddevs,
                            self.means + self.stddevs,
                            alpha=0.2, color=color, label='Std Dev')
        if show_quartiles and self.q1 is not None and self.q3 is not None:
            ax.fill_between(self.distances, self.q1, self.q3,
                            alpha=0.3, color=color, label='Q1-Q3')
        if show_median and self.medians is not None:
            ax.plot(self.distances, self.medians, '--', color=color, label='Median')

        ax.set_xlabel('Distance from track (m)')
        ax.set_ylabel('Elevation (m)')
        ax.legend()
        return fig, ax


class LongitudinalSwath:
    """Class representing a longitudinal swath profile (along-track variation).

    Parameters
    ----------
    means : np.ndarray
        Mean elevation for each sampled track point.
    stddevs : np.ndarray
        Standard deviation of elevation.
    mins : np.ndarray
        Minimum elevation.
    maxs : np.ndarray
        Maximum elevation.
    counts : np.ndarray
        Number of pixels sampled for each point.
    medians : np.ndarray, optional
        Median elevation.
    q1, q3 : np.ndarray, optional
        25th and 75th percentiles.
    percentiles : dict, optional
        Dictionary mapping percentile values to result arrays.
    along_track_distances : np.ndarray, optional
        Cumulative distance of each point along the track.
    track_x, track_y : np.ndarray, optional
        Actual spatial coordinates of the sampled track points.
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
        """Returns a dictionary of statistics for a specific track point index."""
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

    def plot(self, fig=None, ax=None, show_minmax=True, show_std=False,
             show_quartiles=False, show_median=False, **kwargs):
        """Plot the longitudinal swath profile.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
        ax : matplotlib.axes.Axes, optional
        show_minmax : bool, default True
        show_std : bool, default False
        show_quartiles : bool, default False
        show_median : bool, default False
        **kwargs
            Passed to ax.plot.

        Returns
        -------
        fig, ax
        """
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        x = (self.along_track_distances if self.along_track_distances is not None
             else np.arange(len(self.means)))
        label = kwargs.pop('label', 'Mean')
        line = ax.plot(x, self.means, label=label, **kwargs)
        color = line[0].get_color()

        if show_minmax:
            ax.fill_between(x, self.mins, self.maxs,
                            alpha=0.1, color=color, label='Min-Max')
        if show_std:
            ax.fill_between(x, self.means - self.stddevs,
                            self.means + self.stddevs, alpha=0.2,
                            color=color, label='Std Dev')
        if show_quartiles and self.q1 is not None and self.q3 is not None:
            ax.fill_between(x, self.q1, self.q3,
                            alpha=0.3, color=color, label='Q1-Q3')
        if show_median and self.medians is not None:
            ax.plot(x, self.medians, '--', color=color, label='Median')

        ax.set_xlabel('Distance along track (m)')
        ax.set_ylabel('Elevation (m)')
        ax.legend()
        return fig, ax


def _prepare_track(grid, track_x, track_y, input_mode):
    """Internal helper to convert track inputs to (row, col) indices."""
    if input_mode == "indices2D":
        ti = np.array(track_x, dtype=np.float32)
        tj = np.array(track_y, dtype=np.float32)
    elif input_mode == "indices1D":
        ti = (np.asarray(track_x) % grid.rows).astype(np.float32)
        tj = (np.asarray(track_x) // grid.rows).astype(np.float32)
    elif input_mode == "coordinates":
        inv_transform = ~grid.transform
        tx = np.asarray(track_x)
        ty = np.asarray(track_y)
        if grid.bounds is None:
            tj = (tx / grid.cellsize).astype(np.float32)
            ti = (ty / grid.cellsize).astype(np.float32)
        else:
            cols, rows = inv_transform * (tx, ty)
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
    """Ensure a track is D8-contiguous by filling gaps with Bresenham interpolation.

    Rounds track points to integer pixel positions, checks that each consecutive
    pair is D8-adjacent (max(|di|,|dj|) <= 1), and fills any gaps using the
    Bresenham D8 algorithm. Output is in the same format as the input.

    Parameters
    ----------
    grid : GridObject
    track_x, track_y : array-like
    input_mode : str, default "indices2D"

    Returns
    -------
    Same format as input_mode.
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
        count = _swaths.rasterize_path(out_i, out_j, ti_int, tj_int, 0, 0)
        oi = out_i[:count].astype(np.float32)
        oj = out_j[:count].astype(np.float32)

    if input_mode == "indices2D":
        return oi, oj
    if input_mode == "indices1D":
        return (oi.astype(np.intp) + oj.astype(np.intp) * grid.rows).astype(np.intp)
    if input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    return oi, oj


def compute_swath_distance_map(
        grid: GridObject, track_x, track_y=None, half_width: Optional[float] = None,
        input_mode="indices2D", compute_signed=True,
        return_nearest_point=False, return_centre_line=False,
        mask=None) -> Union[GridObject, SwathCentreLine]:
    """Compute a distance map from a polyline track.

    If `half_width` is provided, the map is clipped (NAN outside). If None, a
    full unclipped distance map is computed for all active pixels.

    Parameters
    ----------
    grid : GridObject
        Reference DEM.
    track_x, track_y : array-like
        Track coordinates or indices.
    half_width : float, optional
        Distance limit. If provided, pixels further than this get NAN.
    input_mode : str, default "indices2D"
        Format of track input: "indices2D", "indices1D", or "coordinates".
    compute_signed : bool, default True
        If True, returns signed distance (negative=left, positive=right).
    return_nearest_point : bool, default False
        Include map of nearest track point index in the result.
    return_centre_line : bool, default False
        Calculate medial axis and inward distance (requires `half_width`).
    mask : np.ndarray, optional
        Binary mask for active pixels (only used for full distance maps).

    Returns
    -------
    GridObject or SwathCentreLine
        Distance map grid, or dataclass with additional geometric outputs.
        SwathCentreLine.nearest_point is always populated (needed for longitudinal).
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    hw_px = half_width / grid.cellsize if half_width is not None else float('inf')
    need_signed = compute_signed or return_centre_line

    near_pt_z = np.full(grid.z.shape, -1, dtype=np.intp, order='F')
    best_abs = np.empty(grid.z.shape, dtype=np.float32, order='F')
    signed_dist_px = np.empty(grid.z.shape, dtype=np.float32, order='F') if need_signed else None

    # Build combined mask: caller mask AND NaN exclusion from DEM
    nan_mask = (~np.isnan(grid.z)).astype(np.int8)
    if mask is not None:
        combined_mask = (nan_mask & mask.astype(np.int8)).astype(np.int8)
    else:
        combined_mask = nan_mask

    _swaths.swath_frontier_distance_map(
        best_abs, signed_dist_px, near_pt_z, ti, tj, grid.dims, hw_px,
        combined_mask
    )

    # clipped: outside = beyond radius (FLT_MAX pixels satisfy this too)
    # full:    outside = unvisited (FLT_MAX), hw_px=inf never filters anything
    if half_width is not None:
        outside = best_abs > hw_px
    else:
        outside = best_abs >= np.finfo(np.float32).max

    dist_src = signed_dist_px if (compute_signed and signed_dist_px is not None) else best_abs
    dist_z = np.where(outside, np.nan, dist_src * grid.cellsize).astype(np.float32)
    dist_grid = _grid_from_z(grid, dist_z, "swath_distance")

    near_pt_grid = _grid_from_z(grid, near_pt_z, "swath_nearest_point")

    if not return_centre_line:
        if not return_nearest_point:
            return dist_grid
        return SwathCentreLine(distance_map=dist_grid, nearest_point=near_pt_grid)

    # Centre-line path
    inside = ~outside
    pad = np.pad(inside, 1, constant_values=False)
    boundary = inside & (~pad[:-2, 1:-1] | ~pad[2:, 1:-1] | ~pad[1:-1, :-2] | ~pad[1:-1, 2:])
    pos_seeds = np.flatnonzero((boundary & (signed_dist_px >= 0)).ravel(order='F')).astype(np.intp)
    neg_seeds = np.flatnonzero((boundary & (signed_dist_px <= 0)).ravel(order='F')).astype(np.intp)

    swath_mask = inside.astype(np.int8)
    dist_pos = np.empty(grid.z.shape, dtype=np.float32, order='F')
    dist_neg = np.empty(grid.z.shape, dtype=np.float32, order='F')
    _swaths.swath_boundary_dijkstra(dist_pos, swath_mask, pos_seeds, grid.dims)
    _swaths.swath_boundary_dijkstra(dist_neg, swath_mask, neg_seeds, grid.dims)

    mn = np.minimum(dist_pos, dist_neg)
    dfb_z = np.where(mn >= np.finfo(np.float32).max, np.nan, mn * grid.cellsize).astype(np.float32)

    cli_arr = np.empty(grid.z.size, dtype=np.float32)
    clj_arr = np.empty(grid.z.size, dtype=np.float32)
    cw_arr = np.empty(grid.z.size, dtype=np.float32)
    count = _swaths.voronoi_ridge_to_centreline(
        cli_arr, clj_arr, cw_arr, dist_pos, dist_neg, best_abs, hw_px,
        near_pt_z, ti, tj, grid.dims, grid.cellsize
    )
    count = _swaths.thin_rasterised_line_to_D8(cli_arr, clj_arr, cw_arr, count, grid.dims)

    res = SwathCentreLine(distance_map=dist_grid, nearest_point=near_pt_grid)
    res.dist_from_boundary = _grid_from_z(grid, dfb_z, "swath_dist_from_boundary")
    oi, oj = cli_arr[:count], clj_arr[:count]
    if input_mode == "indices2D":
        res.centre_line_x, res.centre_line_y = oi, oj
    elif input_mode == "indices1D":
        res.centre_line_x = (oi + oj * grid.rows).astype(np.intp)
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        res.centre_line_x, res.centre_line_y = np.array(xs), np.array(ys)
    res.centre_width = cw_arr[:count]
    return res


def transverse_swath(grid: GridObject, distance_map: Union[GridObject, np.ndarray],
                    half_width: float, bin_resolution: float = 10.0,
                    normalize: bool = False,
                    percentiles: Optional[List[int]] = None,
                    custom_stat_fn=None) -> TransverseSwath:
    """Compute a transverse swath profile using a pre-computed signed distance map.

    Aggregates elevations based on their perpendicular distance to the track.

    Parameters
    ----------
    grid : GridObject
        Elevation DEM.
    distance_map : GridObject or np.ndarray
        SIGNED distance map (meters) from `compute_swath_distance_map`.
    half_width : float
        Swath half-width (meters).
    bin_resolution : float, default 10.0
        Spacing between bin centers (meters).
    normalize : bool, default False
        If True, elevations are relative to the mean elevation at the track centre.
    percentiles : list of int, optional
        List of custom percentiles (0-100) to compute for each bin.
    custom_stat_fn : callable, optional
        Function called with the array of elevation values in each bin.
        Signature: fn(values: np.ndarray) -> scalar. Result stored in
        TransverseSwath.custom (NaN for empty bins).

    Returns
    -------
    TransverseSwath
        Aggregated statistics profile.
    """
    dist_arr = (distance_map.z if isinstance(distance_map, GridObject) else distance_map).ravel()
    dem = grid.z.ravel()

    if np.nanmin(dist_arr) >= 0 and np.nanmax(dist_arr) > 0:
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
    """Compute a longitudinal swath profile along a polyline track.

    Aggregates elevations from pixels assigned to each track point via the
    nearest-point map, using a sliding window of ±binning_distance along track.

    Parameters
    ----------
    grid : GridObject
        Elevation DEM.
    track_x, track_y : array-like
        Track coordinates or indices.
    distance_map : GridObject or np.ndarray
        SIGNED distance map (meters) from `compute_swath_distance_map`.
    half_width : float
        Swath half-width (meters).
    binning_distance : float
        Along-track search radius (meters).
    nearest_point : GridObject or np.ndarray
        Per-pixel nearest track-point index from `compute_swath_distance_map`.
    percentiles : list of int, optional
        Custom percentiles to compute.
    input_mode : str, default "indices2D"
        Format of track input.
    skip : int, default 1
        Compute results only for every n-th point to save time.

    Returns
    -------
    LongitudinalSwath
        Profile statistics along the track.
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
    pt_percs: Optional[np.ndarray] = None
    if percentiles is not None:
        perc_list = np.array(percentiles, dtype=np.int32)
        pt_percs = np.zeros((n_out, len(percentiles)), dtype=np.float32)

    res_i = np.zeros(n_out, dtype=np.float32)
    res_j = np.zeros(n_out, dtype=np.float32)

    dist_arr = distance_map.z if isinstance(distance_map, GridObject) else distance_map
    npt_arr = nearest_point.z if isinstance(nearest_point, GridObject) else nearest_point

    full_dist_steps = np.sqrt(np.diff(ti)**2 + np.diff(tj)**2) * grid.cellsize
    full_along_track = np.concatenate(([0.0], np.cumsum(full_dist_steps))).astype(np.float32)

    written = _swaths.swath_longitudinal(
        pt_means, pt_std, pt_min, pt_max, pt_counts,
        pt_medians, pt_q1, pt_q3, perc_list, pt_percs,
        grid.z, ti, tj,
        dist_arr, grid.dims, grid.cellsize, half_width, binning_distance,
        npt_arr, full_along_track, int(skip), res_i, res_j
    )

    res_i = res_i[:written]
    res_j = res_j[:written]
    along_track = full_along_track[::skip][:written]

    perc_dict = None
    if percentiles is not None:
        assert pt_percs is not None
        perc_dict = {p: pt_percs[:written, i] for i, p in enumerate(percentiles)}

    if input_mode == "indices2D":
        track_x_out, track_y_out = res_i, res_j
    elif input_mode == "indices1D":
        track_x_out = res_i.astype(np.intp) + res_j.astype(np.intp) * grid.rows
        track_y_out = None
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (res_j, res_i)
        track_x_out, track_y_out = np.array(xs), np.array(ys)
    else:
        track_x_out, track_y_out = res_i, res_j

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
    """Compute a longitudinal swath profile using an oriented rectangle window.

    For each track point computes a local PCA tangent, defines a rectangle of
    ±binning_distance along-track and ±half_width orthogonal, and accumulates
    stats over all pixels inside. Does not require a pre-computed distance map.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    dem = np.asarray(grid.z, dtype=np.float32)
    nrows, ncols = grid.dims
    cellsize = grid.cellsize
    hw_px = half_width / cellsize
    bd_px = binning_distance / cellsize

    # Precompute PCA tangents for all track points
    n_pts = len(ti)
    half_n = max(1, n_points_regression // 2)
    tang_i = np.empty(n_pts, dtype=np.float32)
    tang_j = np.empty(n_pts, dtype=np.float32)
    for pt in range(n_pts):
        lo = max(0, pt - half_n)
        hi = min(n_pts - 1, pt + half_n)
        if hi - lo < 1:
            lo, hi = (pt, pt + 1) if pt < n_pts - 1 else (pt - 1, pt)
        seg_i = ti[lo:hi + 1]
        seg_j = tj[lo:hi + 1]
        mi, mj = seg_i.mean(), seg_j.mean()
        di, dj = seg_i - mi, seg_j - mj
        cii = float(np.dot(di, di))
        cij = float(np.dot(di, dj))
        cjj = float(np.dot(dj, dj))
        diff = cii - cjj
        D = np.sqrt(diff * diff + 4.0 * cij * cij)
        if cii >= cjj:
            vi, vj = diff + D, 2.0 * cij
        else:
            vi, vj = 2.0 * cij, -diff + D
        vlen = np.sqrt(vi * vi + vj * vj)
        if vlen > 1e-10:
            tang_i[pt] = vi / vlen
            tang_j[pt] = vj / vlen
        else:
            # fallback: finite difference
            d_i = ti[hi] - ti[lo]
            d_j = tj[hi] - tj[lo]
            length = np.sqrt(d_i * d_i + d_j * d_j)
            tang_i[pt] = d_i / length if length > 0 else 1.0
            tang_j[pt] = d_j / length if length > 0 else 0.0

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

        # dem shape is (nrows, ncols); meshgrid uses indexing='ij' so mask shape
        # matches dem[pi_lo:pi_hi+1, pj_lo:pj_hi+1] directly.
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

        sv = np.sort(vals)
        def _pct(p):
            idx = max(0, min(count - 1, int(np.ceil(p / 100.0 * count)) - 1))
            return sv[idx]
        pt_medians[out_idx] = _pct(50)
        pt_q1[out_idx]      = _pct(25)
        pt_q3[out_idx]      = _pct(75)
        if percentiles is not None and pt_percs is not None:
            for p_idx, p in enumerate(percentiles):
                pt_percs[out_idx, p_idx] = _pct(p)

    # Along-track distance
    full_dist_steps = np.sqrt(np.diff(ti)**2 + np.diff(tj)**2) * cellsize
    full_along_track = np.concatenate(([0.0], np.cumsum(full_dist_steps)))
    along_track = full_along_track[::skip][:n_out]

    perc_dict = None
    if percentiles is not None and pt_percs is not None:
        perc_dict = {p: pt_percs[:, i] for i, p in enumerate(percentiles)}

    track_i_out = ti[::skip][:n_out]
    track_j_out = tj[::skip][:n_out]

    if input_mode == "indices2D":
        track_x_out, track_y_out = track_i_out, track_j_out
    elif input_mode == "indices1D":
        track_x_out = (track_i_out.astype(np.intp) +
                       track_j_out.astype(np.intp) * grid.rows)
        track_y_out = None
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (track_j_out, track_i_out)
        track_x_out, track_y_out = np.array(xs), np.array(ys)
    else:
        track_x_out, track_y_out = track_i_out, track_j_out

    return LongitudinalSwath(pt_means, pt_std, pt_min, pt_max, pt_counts,
                             pt_medians, pt_q1, pt_q3, perc_dict, along_track,
                             track_x_out, track_y_out)


def get_point_pixels(grid: GridObject, track_x, track_y,
                    distance_map: Union[GridObject, np.ndarray],
                    point_index: int, half_width: float,
                    binning_distance: float,
                    nearest_point,
                    input_mode: str = "indices2D"):
    """Retrieve pixel indices or coordinates assigned to a single track point.

    Mirrors the pixel selection logic used in `longitudinal_swath`.

    Parameters
    ----------
    grid : GridObject
    track_x, track_y : array-like
    distance_map : GridObject or np.ndarray
        SIGNED distance map.
    point_index : int
        Index of the track point.
    half_width : float
    binning_distance : float
    nearest_point : GridObject or np.ndarray
        Per-pixel nearest track-point index.
    input_mode : str

    Returns
    -------
    tuple of np.ndarray
        Pixel coordinates/indices in the format matching `input_mode`.
    """
    dist_arr = distance_map.z if isinstance(distance_map, GridObject) else distance_map
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    npt_arr = nearest_point.z if isinstance(nearest_point, GridObject) else nearest_point

    cum_dist = np.concatenate(([0.0], np.cumsum(
        np.sqrt(np.diff(ti)**2 + np.diff(tj)**2) * grid.cellsize
    ))).astype(np.float32)

    pi = np.zeros(grid.z.size, dtype=np.intp)
    pj = np.zeros(grid.z.size, dtype=np.intp)

    count = _swaths.swath_get_point_pixels(
        pi, pj, ti, tj, point_index, dist_arr, grid.dims,
        grid.cellsize, half_width, binning_distance, npt_arr, cum_dist
    )

    oi = pi[:count]
    oj = pj[:count]

    if input_mode == "indices2D":
        return oi, oj
    if input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    if input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    return oi, oj


def get_windowed_point_samples(grid: GridObject, track_x, track_y,
                               point_index: int, half_width: float,
                               binning_distance: float,
                               n_points_regression: int = 5,
                               input_mode: str = "indices2D"):
    """Retrieve pixel indices or coordinates inside a specific window.

    Mirrors the oriented-rectangle selection used in `longitudinal_swath_windowed`.

    Parameters
    ----------
    grid : GridObject
    track_x, track_y : array-like
    point_index : int
    half_width : float
    binning_distance : float
    n_points_regression : int
    input_mode : str

    Returns
    -------
    tuple of np.ndarray
        Pixel coordinates/indices in the format matching `input_mode`.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    n_pts = len(ti)
    hw_px = half_width / grid.cellsize
    bd_px = binning_distance / grid.cellsize
    nrows, ncols = grid.dims

    # PCA tangent at point_index
    half_n = max(1, n_points_regression // 2)
    lo = max(0, point_index - half_n)
    hi = min(n_pts - 1, point_index + half_n)
    if hi - lo < 1:
        lo, hi = (point_index, point_index + 1) if point_index < n_pts - 1 else (point_index - 1, point_index)
    seg_i, seg_j = ti[lo:hi + 1], tj[lo:hi + 1]
    mi, mj = seg_i.mean(), seg_j.mean()
    di, dj = seg_i - mi, seg_j - mj
    cii, cij, cjj = float(np.dot(di, di)), float(np.dot(di, dj)), float(np.dot(dj, dj))
    diff = cii - cjj
    D = np.sqrt(diff * diff + 4.0 * cij * cij)
    vi, vj = (diff + D, 2.0 * cij) if cii >= cjj else (2.0 * cij, -diff + D)
    vlen = np.sqrt(vi * vi + vj * vj)
    if vlen > 1e-10:
        t_i, t_j = vi / vlen, vj / vlen
    else:
        d_i, d_j = ti[hi] - ti[lo], tj[hi] - tj[lo]
        length = np.sqrt(d_i * d_i + d_j * d_j)
        t_i, t_j = (d_i / length, d_j / length) if length > 0 else (1.0, 0.0)
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

    if input_mode == "indices2D":
        return oi, oj
    if input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    if input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    return oi, oj


def rasterize_path(grid: GridObject, track_x, track_y=None,
                               input_mode="indices2D", close_loop=False,
                               use_d4=False):
    """Rasterize a continuous path between ordered reference points.

    Uses Bresenham's line algorithm to connect consecutive points.

    Parameters
    ----------
    grid : GridObject
    track_x, track_y : array-like
        Reference track points.
    input_mode : str, default "indices2D"
    close_loop : bool, default False
        Connect the last point back to the first.
    use_d4 : bool, default False
        If True, uses D4 connectivity; otherwise D8.

    Returns
    -------
    tuple of np.ndarray
        Rasterized path coordinates/indices.
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

    count = _swaths.rasterize_path(out_i, out_j, ti_int, tj_int,
                                               int(close_loop), int(use_d4))

    oi = out_i[:count]
    oj = out_j[:count]

    if input_mode == "indices2D":
        return oi, oj
    if input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    if input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    return oi, oj


def simplify_line(grid: GridObject, track_x, track_y=None, tolerance: float = 1.0,
                  method: int = 0, input_mode: str = "indices2D"):
    """Simplify a polyline using the Iterative End-Point Fit (IEF) engine.

    Reduces vertices while preserving shape using various stopping criteria.

    Parameters:
    -----------
    grid : GridObject
        Reference grid.
    track_x, track_y : array-like
        Track coordinates or indices.
    tolerance : float
        Meaning depends on method:
        - method 0 (FIXED_N): exact number of output points (clamped to [2, n_points]).
        - method 1 (KNEEDLE): ignored (automatic knee detection).
        - method 2 (VW_AREA): triangle area threshold (coordinate units squared).
    method : int
        Simplification method (0-2).
    input_mode : str, default "indices2D"
        "indices2D", "indices1D" or "coordinates".

    Returns
    -------
    tuple of np.ndarray
        Simplified track coordinates/indices.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    n_points = len(ti)

    out_i = np.zeros(n_points, dtype=np.float32)
    out_j = np.zeros(n_points, dtype=np.float32)

    count = _swaths.simplify_line(out_i, out_j, ti, tj, tolerance, method)

    oi = out_i[:count]
    oj = out_j[:count]

    if input_mode == "indices2D":
        return oi, oj
    if input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    if input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    return oi, oj
