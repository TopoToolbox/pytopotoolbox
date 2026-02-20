"""
This module provides the Python interface for swath profile analysis.
It includes classes and functions for computing and plotting transverse
and longitudinal swath profiles from Digital Elevation Models (DEMs).
"""
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict
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
    nearest_segment : GridObject, optional
        Map of nearest track segment index for each pixel.
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
    nearest_segment: Optional[GridObject] = None
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
                 medians=None, q1=None, q3=None, percentiles=None):
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
            'distance': self.along_track_distances[idx] if self.along_track_distances is not None else idx,
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

        x = self.along_track_distances if self.along_track_distances is not None else np.arange(len(self.means))
        label = kwargs.pop('label', 'Mean')
        line = ax.plot(x, self.means, label=label, **kwargs)
        color = line[0].get_color()

        if show_minmax:
            ax.fill_between(x, self.mins, self.maxs,
                            alpha=0.1, color=color, label='Min-Max')
        if show_std:
            ax.fill_between(x, self.means - self.stddevs,
                            self.means + self.stddevs, alpha=0.2, color=color, label='Std Dev')
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
        ti = (track_x % grid.rows).astype(np.float32)
        tj = (track_x // grid.rows).astype(np.float32)
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


def compute_swath_distance_map(
        grid: GridObject, track_x, track_y=None, half_width: Optional[float] = None,
        input_mode="indices2D", compute_signed=True,
        return_nearest_segment=False, return_centre_line=False, mask=None) -> Union[GridObject, SwathCentreLine]:
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
    return_nearest_segment : bool, default False
        Include map of nearest track segment index.
    return_centre_line : bool, default False
        Calculate medial axis and inward distance (requires `half_width`).
    mask : np.ndarray, optional
        Binary mask for active pixels (only used for full distance maps).

    Returns
    -------
    GridObject or SwathCentreLine
        Distance map grid, or dataclass with additional geometric outputs.
    """
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)
    dist_z = np.full(grid.z.shape, np.nan, dtype=np.float32, order='F')
    near_seg_z = None
    if return_nearest_segment or return_centre_line:
        near_seg_z = np.full(grid.z.shape, -1, dtype=np.intp, order='F')

    if half_width is not None:
        # Clipped distance map
        dfb_z = None
        cli_arr = None
        clj_arr = None
        cw_arr = None
        if return_centre_line:
            dfb_z = np.full(grid.z.shape, np.nan, dtype=np.float32, order='F')
            cli_arr = np.zeros(grid.z.size, dtype=np.float32)
            clj_arr = np.zeros(grid.z.size, dtype=np.float32)
            cw_arr = np.zeros(grid.z.size, dtype=np.float32)

        count = _swaths.swath_compute_distance_map(
            dist_z, near_seg_z, dfb_z, cli_arr, clj_arr, cw_arr,
            ti, tj, grid.dims, grid.cellsize, half_width, int(compute_signed)
        )

        dist_grid = _grid_from_z(grid, dist_z, "swath_distance")
        if not return_centre_line and not return_nearest_segment:
            return dist_grid

        # Build dataclass
        res = SwathCentreLine(distance_map=dist_grid)
        if return_nearest_segment:
            res.nearest_segment = _grid_from_z(grid, near_seg_z, "swath_nearest_segment")
        if return_centre_line:
            res.dist_from_boundary = _grid_from_z(grid, dfb_z, "swath_dist_from_boundary")
            
            oi = cli_arr[:count]
            oj = clj_arr[:count]
            if input_mode == "indices2D":
                res.centre_line_x, res.centre_line_y = oi, oj
            elif input_mode == "indices1D":
                res.centre_line_x = (oi + oj * grid.rows).astype(np.intp)
            elif input_mode == "coordinates":
                xs, ys = grid.transform * (oj, oi)
                res.centre_line_x, res.centre_line_y = np.array(xs), np.array(ys)
            
            res.centre_width = cw_arr[:count]
        return res

    else:
        # Full distance map
        _swaths.swath_compute_full_distance_map(
            dist_z, near_seg_z, ti, tj, grid.dims, grid.cellsize, grid.z, mask, int(compute_signed)
        )
        dist_grid = _grid_from_z(grid, dist_z, "swath_full_distance")
        if return_nearest_segment:
            return SwathCentreLine(distance_map=dist_grid,
                                   nearest_segment=_grid_from_z(grid, near_seg_z, "swath_nearest_segment"))
        return dist_grid


def transverse_swath(grid: GridObject, distance_map: Union[GridObject, np.ndarray],
                    half_width: float, bin_resolution: float = 10.0,
                    normalize: bool = False, percentiles: Optional[List[int]] = None) -> TransverseSwath:
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
        If True, elevations are relative to the nearest track point elevation.
    percentiles : list of int, optional
        List of custom percentiles (0-100) to compute for each bin.

    Returns
    -------
    TransverseSwath
        Aggregated statistics profile.
    """
    dist_arr = distance_map.z if isinstance(distance_map, GridObject) else distance_map
    if np.nanmin(dist_arr) >= 0 and np.nanmax(dist_arr) > 0:
        import warnings
        warnings.warn("Distance map appears to be absolute (unsigned). Results will only cover positive side.")

    n_bins = _swaths.swath_compute_nbins(half_width, bin_resolution)
    bin_dist = np.zeros(n_bins, dtype=np.float32)
    bin_means = np.zeros(n_bins, dtype=np.float32)
    bin_std = np.zeros(n_bins, dtype=np.float32)
    bin_min = np.zeros(n_bins, dtype=np.float32)
    bin_max = np.zeros(n_bins, dtype=np.float32)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_medians = np.zeros(n_bins, dtype=np.float32)
    bin_q1 = np.zeros(n_bins, dtype=np.float32)
    bin_q3 = np.zeros(n_bins, dtype=np.float32)

    perc_list = None
    bin_percs = None
    if percentiles is not None:
        perc_list = np.array(percentiles, dtype=np.int32)
        bin_percs = np.zeros((n_bins, len(percentiles)), dtype=np.float32)

    _swaths.swath_transverse(
        bin_dist, bin_means, bin_std, bin_min, bin_max, bin_counts,
        bin_medians, bin_q1, bin_q3, perc_list, bin_percs,
        grid.z, dist_arr, grid.dims, half_width, bin_resolution, int(normalize)
    )

    perc_dict = {p: bin_percs[:, i] for i, p in enumerate(percentiles)} if percentiles else None

    return TransverseSwath(bin_dist, bin_means, bin_std, bin_min, bin_max, bin_counts,
                           bin_medians, bin_q1, bin_q3, perc_dict)


def longitudinal_swath(grid: GridObject, track_x, track_y, distance_map: Union[GridObject, np.ndarray],
                      half_width: float, binning_distance: float = 0.0,
                      n_points_regression: int = 5, use_segment_seeds: bool = True,
                      percentiles: Optional[List[int]] = None,
                      input_mode: str = "indices2D", skip: int = 1) -> LongitudinalSwath:
    """Compute a longitudinal swath profile along a polyline track.

    Aggregates elevations assigned to each track point based on perpendicular cross-sections.

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
    binning_distance : float, default 0.0
        Along-track search radius (meters). If > 0, gathers pixels within a
        bounding box between orthogonals.
    n_points_regression : int, default 5
        Neighbourhood size for local track tangent estimation via PCA.
    use_segment_seeds : bool, default True
        Use high-precision segment-based assignment (EDT) instead of point-based.
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
    pt_percs = None
    if percentiles is not None:
        perc_list = np.array(percentiles, dtype=np.int32)
        pt_percs = np.zeros((n_out, len(percentiles)), dtype=np.float32)

    res_i = np.zeros(n_out, dtype=np.float32)
    res_j = np.zeros(n_out, dtype=np.float32)

    dist_arr = distance_map.z if isinstance(distance_map, GridObject) else distance_map

    written = _swaths.swath_longitudinal(
        pt_means, pt_std, pt_min, pt_max, pt_counts,
        pt_medians, pt_q1, pt_q3, perc_list, pt_percs,
        grid.z, ti, tj,
        dist_arr, grid.dims, grid.cellsize, half_width, binning_distance,
        int(n_points_regression), int(use_segment_seeds), int(skip), res_i, res_j
    )

    res_i = res_i[:written]
    res_j = res_j[:written]

    full_dist_steps = np.sqrt(np.diff(ti)**2 + np.diff(tj)**2) * grid.cellsize
    full_along_track = np.concatenate(([0], np.cumsum(full_dist_steps)))
    along_track = full_along_track[::skip][:written]

    perc_dict = {p: pt_percs[:written, i] for i, p in enumerate(percentiles)} if percentiles else None

    if input_mode == "indices2D":
        track_x_out, track_y_out = res_i, res_j
    elif input_mode == "indices1D":
        track_x_out = (res_i.astype(np.intp) + res_j.astype(np.intp) * grid.rows)
        track_y_out = None
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (res_j, res_i)
        track_x_out, track_y_out = np.array(xs), np.array(ys)
    else:
        track_x_out, track_y_out = res_i, res_j

    return LongitudinalSwath(pt_means[:written], pt_std[:written], pt_min[:written], pt_max[:written], pt_counts[:written],
                             pt_medians[:written], pt_q1[:written], pt_q3[:written], perc_dict, along_track,
                             track_x_out, track_y_out)


def longitudinal_swath_windowed(grid: GridObject, track_x, track_y,
                               half_width: float, binning_distance: float,
                               n_points_regression: int = 5,
                               percentiles: Optional[List[int]] = None,
                               input_mode: str = "indices2D", skip: int = 1) -> LongitudinalSwath:
    """Compute a longitudinal swath profile using sliding-window statistics.

    Unlike `longitudinal_swath`, this method samples every pixel inside an
    oriented rectangle window centered on each track point. It uses a faster
    histogram-based percentile algorithm and does NOT require a pre-computed
    distance map.

    Parameters
    ----------
    grid : GridObject
        Elevation DEM.
    track_x, track_y : array-like
        Track coordinates or indices.
    half_width : float
        Rectangle half-width (orthogonal distance, meters).
    binning_distance : float
        Rectangle half-length (along-track window radius, meters).
    n_points_regression : int, default 5
        Neighbourhood size for PCA tangent estimation.
    percentiles : list of int, optional
        Custom percentiles to compute.
    input_mode : str, default "indices2D"
        Format of track input.
    skip : int, default 1
        Sample every n-th track point.

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
    pt_percs = None
    if percentiles is not None:
        perc_list = np.array(percentiles, dtype=np.int32)
        pt_percs = np.zeros((n_out, len(percentiles)), dtype=np.float32)

    res_i = np.zeros(n_out, dtype=np.float32)
    res_j = np.zeros(n_out, dtype=np.float32)

    written = _swaths.swath_longitudinal_windowed(
        pt_means, pt_std, pt_min, pt_max, pt_counts,
        pt_medians, pt_q1, pt_q3, perc_list, pt_percs,
        grid.z, ti, tj,
        grid.dims, grid.cellsize, half_width, binning_distance,
        int(n_points_regression), int(skip), res_i, res_j
    )

    res_i = res_i[:written]
    res_j = res_j[:written]

    full_dist_steps = np.sqrt(np.diff(ti)**2 + np.diff(tj)**2) * grid.cellsize
    full_along_track = np.concatenate(([0], np.cumsum(full_dist_steps)))
    along_track = full_along_track[::skip][:written]

    perc_dict = {p: pt_percs[:written, i] for i, p in enumerate(percentiles)} if percentiles else None

    if input_mode == "indices2D":
        track_x_out, track_y_out = res_i, res_j
    elif input_mode == "indices1D":
        track_x_out = (res_i.astype(np.intp) + res_j.astype(np.intp) * grid.rows)
        track_y_out = None
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (res_j, res_i)
        track_x_out, track_y_out = np.array(xs), np.array(ys)
    else:
        track_x_out, track_y_out = res_i, res_j

    return LongitudinalSwath(pt_means[:written], pt_std[:written], pt_min[:written], pt_max[:written], pt_counts[:written],
                             pt_medians[:written], pt_q1[:written], pt_q3[:written], perc_dict, along_track,
                             track_x_out, track_y_out)


def get_point_pixels(grid: GridObject, track_x, track_y, distance_map: Union[GridObject, np.ndarray],
                    point_index: int, half_width: float, binning_distance: float = 0.0,
                    n_points_regression: int = 5, use_segment_seeds: bool = True,
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
    n_points_regression : int
    use_segment_seeds : bool
    input_mode : str

    Returns
    -------
    tuple of np.ndarray
        Pixel coordinates/indices in the format matching `input_mode`.
    """
    dist_arr = distance_map.z if isinstance(distance_map, GridObject) else distance_map
    ti, tj = _prepare_track(grid, track_x, track_y, input_mode)

    pi = np.zeros(grid.z.size, dtype=np.intp)
    pj = np.zeros(grid.z.size, dtype=np.intp)

    count = _swaths.swath_get_point_pixels(
        pi, pj, ti, tj, point_index, dist_arr, grid.dims,
        grid.cellsize, half_width, binning_distance, int(n_points_regression), int(use_segment_seeds)
    )
    
    oi = pi[:count]
    oj = pj[:count]
    
    if input_mode == "indices2D":
        return oi, oj
    elif input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    else:
        return oi, oj


def get_windowed_point_samples(grid: GridObject, track_x, track_y,
                               point_index: int, half_width: float, binning_distance: float,
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

    pi = np.zeros(grid.z.size, dtype=np.intp)
    pj = np.zeros(grid.z.size, dtype=np.intp)

    count = _swaths.swath_windowed_get_point_samples(
        pi, pj, ti, tj, point_index, grid.dims,
        grid.cellsize, half_width, binning_distance, int(n_points_regression)
    )
    
    oi = pi[:count]
    oj = pj[:count]
    
    if input_mode == "indices2D":
        return oi, oj
    elif input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    else:
        return oi, oj


def sample_points_between_refs(grid: GridObject, track_x, track_y=None, 
                               input_mode="indices2D", close_loop=False, use_d4=False):
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
        max_size += int(max(np.abs(ti_int[-1] - ti_int[0]), np.abs(tj_int[-1] - tj_int[0])) + 1)
    
    if use_d4:
        max_size *= 2
        
    out_i = np.zeros(max_size, dtype=np.intp)
    out_j = np.zeros(max_size, dtype=np.intp)
    
    count = _swaths.sample_points_between_refs(out_i, out_j, ti_int, tj_int, int(close_loop), int(use_d4))
    
    oi = out_i[:count]
    oj = out_j[:count]
    
    if input_mode == "indices2D":
        return oi, oj
    elif input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    else:
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
        - method 2 (AIC): RMSE noise floor (coordinate units). Larger tolerance -> fewer points.
        - method 3 (BIC): RMSE noise floor (same as AIC).
        - method 4 (MDL): RMSE noise floor (same as AIC).
        - method 5 (VW_AREA): triangle area threshold (coordinate units squared).
        - method 6 (L_METHOD): ignored (L-method elbow detection).
    method : int
        Simplification method (0-6). Use SIMPLIFY_* constants.
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
    elif input_mode == "indices1D":
        return (oi + oj * grid.rows).astype(np.intp)
    elif input_mode == "coordinates":
        xs, ys = grid.transform * (oj, oi)
        return np.array(xs), np.array(ys)
    else:
        return oi, oj
