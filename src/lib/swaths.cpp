// This file contains bindings for the swath profile functions

// It is expected, that the #include statements can raise errors in your IDE.
// The needed files for those imports are only provided during the build process.

extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib>

namespace py = pybind11;

// Wraps the swath_compute_nbins helper function
ptrdiff_t wrap_swath_compute_nbins(float half_width, float bin_resolution){
    return swath_compute_nbins(half_width, bin_resolution);
}

// Wraps swath_compute_distance_map
ptrdiff_t wrap_swath_compute_distance_map(
        py::array_t<float> distance_from_track,
        py::object nearest_segment_obj,
        py::object dist_from_boundary_obj,
        py::object centre_line_i_obj,
        py::object centre_line_j_obj,
        py::object centre_width_obj,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize,
        float half_width,
        int compute_signed){

    float *dist_ptr = distance_from_track.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    ptrdiff_t *near_seg_ptr = nearest_segment_obj.is_none() ? nullptr : nearest_segment_obj.cast<py::array_t<ptrdiff_t>>().mutable_data();
    float *dfb_ptr = dist_from_boundary_obj.is_none() ? nullptr : dist_from_boundary_obj.cast<py::array_t<float>>().mutable_data();
    float *cli_ptr = centre_line_i_obj.is_none() ? nullptr : centre_line_i_obj.cast<py::array_t<float>>().mutable_data();
    float *clj_ptr = centre_line_j_obj.is_none() ? nullptr : centre_line_j_obj.cast<py::array_t<float>>().mutable_data();
    float *cw_ptr = centre_width_obj.is_none() ? nullptr : centre_width_obj.cast<py::array_t<float>>().mutable_data();

    return swath_compute_distance_map(dist_ptr, near_seg_ptr, dfb_ptr, cli_ptr, clj_ptr, cw_ptr,
                                       track_i_ptr, track_j_ptr, n_track_points, dims_ptr,
                                       cellsize, half_width, compute_signed);
}

// Wraps swath_compute_full_distance_map
void wrap_swath_compute_full_distance_map(
        py::array_t<float> distance,
        py::object nearest_segment_obj,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize,
        py::object dem_obj,
        py::object mask_obj,
        int compute_signed){

    float *dist_ptr = distance.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    ptrdiff_t *near_seg_ptr = nearest_segment_obj.is_none() ? nullptr : nearest_segment_obj.cast<py::array_t<ptrdiff_t>>().mutable_data();
    float *dem_ptr = dem_obj.is_none() ? nullptr : dem_obj.cast<py::array_t<float>>().mutable_data();
    int8_t *mask_ptr = mask_obj.is_none() ? nullptr : mask_obj.cast<py::array_t<int8_t>>().mutable_data();

    swath_compute_full_distance_map(dist_ptr, near_seg_ptr, track_i_ptr, track_j_ptr,
                                     n_track_points, dims_ptr, cellsize, dem_ptr, mask_ptr, compute_signed);
}

// Wraps swath_transverse
void wrap_swath_transverse(
        py::array_t<float> bin_distances,
        py::array_t<float> bin_means,
        py::array_t<float> bin_stddevs,
        py::array_t<float> bin_mins,
        py::array_t<float> bin_maxs,
        py::array_t<ptrdiff_t> bin_counts,
        py::object bin_medians_obj,
        py::object bin_q1_obj,
        py::object bin_q3_obj,
        py::object percentile_list_obj,
        py::object bin_percentiles_obj,
        py::array_t<float> dem,
        py::array_t<float> distance_from_track,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float half_width,
        float bin_resolution,
        int normalize){

    float *bin_dist_ptr = bin_distances.mutable_data();
    float *bin_means_ptr = bin_means.mutable_data();
    float *bin_std_ptr = bin_stddevs.mutable_data();
    float *bin_min_ptr = bin_mins.mutable_data();
    float *bin_max_ptr = bin_maxs.mutable_data();
    ptrdiff_t *bin_counts_ptr = bin_counts.mutable_data();

    float *bin_medians_ptr = bin_medians_obj.is_none() ? nullptr : bin_medians_obj.cast<py::array_t<float>>().mutable_data();
    float *bin_q1_ptr = bin_q1_obj.is_none() ? nullptr : bin_q1_obj.cast<py::array_t<float>>().mutable_data();
    float *bin_q3_ptr = bin_q3_obj.is_none() ? nullptr : bin_q3_obj.cast<py::array_t<float>>().mutable_data();
    
    int *perc_list_ptr = nullptr;
    ptrdiff_t n_percs = 0;
    if (!percentile_list_obj.is_none()) {
        auto perc_list = percentile_list_obj.cast<py::array_t<int>>();
        perc_list_ptr = perc_list.mutable_data();
        n_percs = perc_list.size();
    }
    float *bin_percs_ptr = bin_percentiles_obj.is_none() ? nullptr : bin_percentiles_obj.cast<py::array_t<float>>().mutable_data();

    float *dem_ptr = dem.mutable_data();
    float *dist_map_ptr = distance_from_track.mutable_data();
    ptrdiff_t n_bins = bin_distances.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    swath_transverse(bin_dist_ptr, bin_means_ptr, bin_std_ptr, bin_min_ptr, bin_max_ptr, bin_counts_ptr,
                      bin_medians_ptr, bin_q1_ptr, bin_q3_ptr, perc_list_ptr, n_percs, bin_percs_ptr,
                      dem_ptr, dist_map_ptr, dims_ptr, half_width, bin_resolution, n_bins, normalize);
}

// Wraps swath_longitudinal
ptrdiff_t wrap_swath_longitudinal(
        py::array_t<float> point_means,
        py::array_t<float> point_stddevs,
        py::array_t<float> point_mins,
        py::array_t<float> point_maxs,
        py::array_t<ptrdiff_t> point_counts,
        py::object point_medians_obj,
        py::object point_q1_obj,
        py::object point_q3_obj,
        py::object percentile_list_obj,
        py::object point_percentiles_obj,
        py::array_t<float> dem,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        py::array_t<float> distance_from_track,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize,
        float half_width,
        float binning_distance,
        ptrdiff_t n_points_regression,
        ptrdiff_t use_segment_seeds,
        ptrdiff_t skip,
        py::object result_track_i_obj,
        py::object result_track_j_obj){

    float *means_ptr = point_means.mutable_data();
    float *std_ptr = point_stddevs.mutable_data();
    float *min_ptr = point_mins.mutable_data();
    float *max_ptr = point_maxs.mutable_data();
    ptrdiff_t *counts_ptr = point_counts.mutable_data();

    float *medians_ptr = point_medians_obj.is_none() ? nullptr : point_medians_obj.cast<py::array_t<float>>().mutable_data();
    float *q1_ptr = point_q1_obj.is_none() ? nullptr : point_q1_obj.cast<py::array_t<float>>().mutable_data();
    float *q3_ptr = point_q3_obj.is_none() ? nullptr : point_q3_obj.cast<py::array_t<float>>().mutable_data();
    
    int *perc_list_ptr = nullptr;
    ptrdiff_t n_percs = 0;
    if (!percentile_list_obj.is_none()) {
        auto perc_list = percentile_list_obj.cast<py::array_t<int>>();
        perc_list_ptr = perc_list.mutable_data();
        n_percs = perc_list.size();
    }
    float *point_percs_ptr = point_percentiles_obj.is_none() ? nullptr : point_percentiles_obj.cast<py::array_t<float>>().mutable_data();

    float *dem_ptr = dem.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();
    float *dist_map_ptr = distance_from_track.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    float *res_i_ptr = result_track_i_obj.is_none() ? nullptr : result_track_i_obj.cast<py::array_t<float>>().mutable_data();
    float *res_j_ptr = result_track_j_obj.is_none() ? nullptr : result_track_j_obj.cast<py::array_t<float>>().mutable_data();

    return swath_longitudinal(means_ptr, std_ptr, min_ptr, max_ptr, counts_ptr,
                               medians_ptr, q1_ptr, q3_ptr, perc_list_ptr, n_percs, point_percs_ptr,
                               dem_ptr, track_i_ptr, track_j_ptr, n_track_points,
                               dist_map_ptr, dims_ptr, cellsize, half_width, binning_distance,
                               n_points_regression, use_segment_seeds, skip, res_i_ptr, res_j_ptr);
}

// Wraps swath_longitudinal_windowed
ptrdiff_t wrap_swath_longitudinal_windowed(
        py::array_t<float> point_means,
        py::array_t<float> point_stddevs,
        py::array_t<float> point_mins,
        py::array_t<float> point_maxs,
        py::array_t<ptrdiff_t> point_counts,
        py::object point_medians_obj,
        py::object point_q1_obj,
        py::object point_q3_obj,
        py::object percentile_list_obj,
        py::object point_percentiles_obj,
        py::array_t<float> dem,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize,
        float half_width,
        float binning_distance,
        ptrdiff_t n_points_regression,
        ptrdiff_t skip,
        py::object result_track_i_obj,
        py::object result_track_j_obj){

    float *means_ptr = point_means.mutable_data();
    float *std_ptr = point_stddevs.mutable_data();
    float *min_ptr = point_mins.mutable_data();
    float *max_ptr = point_maxs.mutable_data();
    ptrdiff_t *counts_ptr = point_counts.mutable_data();

    float *medians_ptr = point_medians_obj.is_none() ? nullptr : point_medians_obj.cast<py::array_t<float>>().mutable_data();
    float *q1_ptr = point_q1_obj.is_none() ? nullptr : point_q1_obj.cast<py::array_t<float>>().mutable_data();
    float *q3_ptr = point_q3_obj.is_none() ? nullptr : point_q3_obj.cast<py::array_t<float>>().mutable_data();
    
    int *perc_list_ptr = nullptr;
    ptrdiff_t n_percs = 0;
    if (!percentile_list_obj.is_none()) {
        auto perc_list = percentile_list_obj.cast<py::array_t<int>>();
        perc_list_ptr = perc_list.mutable_data();
        n_percs = perc_list.size();
    }
    float *point_percs_ptr = point_percentiles_obj.is_none() ? nullptr : point_percentiles_obj.cast<py::array_t<float>>().mutable_data();

    float *dem_ptr = dem.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    float *res_i_ptr = result_track_i_obj.is_none() ? nullptr : result_track_i_obj.cast<py::array_t<float>>().mutable_data();
    float *res_j_ptr = result_track_j_obj.is_none() ? nullptr : result_track_j_obj.cast<py::array_t<float>>().mutable_data();

    return swath_longitudinal_windowed(means_ptr, std_ptr, min_ptr, max_ptr, counts_ptr,
                                        medians_ptr, q1_ptr, q3_ptr, perc_list_ptr, n_percs, point_percs_ptr,
                                        dem_ptr, track_i_ptr, track_j_ptr, n_track_points,
                                        dims_ptr, cellsize, half_width, binning_distance, n_points_regression,
                                        skip, res_i_ptr, res_j_ptr);
}

// Wraps swath_windowed_get_point_samples
ptrdiff_t wrap_swath_windowed_get_point_samples(
        py::array_t<ptrdiff_t> pixels_i,
        py::array_t<ptrdiff_t> pixels_j,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        ptrdiff_t point_index,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize,
        float half_width,
        float binning_distance,
        ptrdiff_t n_points_regression){

    ptrdiff_t *pi_ptr = pixels_i.mutable_data();
    ptrdiff_t *pj_ptr = pixels_j.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    return swath_windowed_get_point_samples(pi_ptr, pj_ptr, track_i_ptr, track_j_ptr, n_track_points,
                                            point_index, dims_ptr, cellsize, half_width,
                                            binning_distance, n_points_regression);
}

// Wraps swath_get_point_pixels
ptrdiff_t wrap_swath_get_point_pixels(
        py::array_t<ptrdiff_t> pixels_i,
        py::array_t<ptrdiff_t> pixels_j,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        ptrdiff_t point_index,
        py::array_t<float> distance_from_track,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize,
        float half_width,
        float binning_distance,
        ptrdiff_t n_points_regression,
        ptrdiff_t use_segment_seeds){

    ptrdiff_t *pi_ptr = pixels_i.mutable_data();
    ptrdiff_t *pj_ptr = pixels_j.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();
    float *dist_map_ptr = distance_from_track.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    return swath_get_point_pixels(pi_ptr, pj_ptr, track_i_ptr, track_j_ptr, n_track_points,
                                   point_index, dist_map_ptr, dims_ptr, cellsize, half_width,
                                   binning_distance, n_points_regression, use_segment_seeds);
}

// wrap_sample_points_between_refs
ptrdiff_t wrap_sample_points_between_refs(
        py::array_t<ptrdiff_t> out_i,
        py::array_t<ptrdiff_t> out_j,
        py::array_t<ptrdiff_t> ref_i,
        py::array_t<ptrdiff_t> ref_j,
        int close_loop,
        int use_d4){

    ptrdiff_t *oi_ptr = out_i.mutable_data();
    ptrdiff_t *oj_ptr = out_j.mutable_data();
    ptrdiff_t *ri_ptr = ref_i.mutable_data();
    ptrdiff_t *rj_ptr = ref_j.mutable_data();
    ptrdiff_t n_refs = ref_i.size();

    return sample_points_between_refs(oi_ptr, oj_ptr, ri_ptr, rj_ptr, n_refs, close_loop, use_d4);
}

// wrap_simplify_line
ptrdiff_t wrap_simplify_line(
        py::array_t<float> out_i,
        py::array_t<float> out_j,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        float tolerance,
        int method){

    float *oi_ptr = out_i.mutable_data();
    float *oj_ptr = out_j.mutable_data();
    float *ti_ptr = track_i.mutable_data();
    float *tj_ptr = track_j.mutable_data();
    ptrdiff_t n_points = track_i.size();

    return simplify_line(oi_ptr, oj_ptr, ti_ptr, tj_ptr, n_points, tolerance, method);
}

PYBIND11_MODULE(_swaths, m) {
    m.doc() = "Swath profile analysis functions for libtopotoolbox";

    m.def("swath_compute_nbins", &wrap_swath_compute_nbins,
          "Compute number of bins for swath profile",
          py::arg("half_width"),
          py::arg("bin_resolution"));

    m.def("swath_compute_distance_map", &wrap_swath_compute_distance_map,
          "Compute clipped distance map from track with optional centre-line",
          py::arg("distance_from_track"),
          py::arg("nearest_segment") = py::none(),
          py::arg("dist_from_boundary") = py::none(),
          py::arg("centre_line_i") = py::none(),
          py::arg("centre_line_j") = py::none(),
          py::arg("centre_width") = py::none(),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("dims"),
          py::arg("cellsize"),
          py::arg("half_width"),
          py::arg("compute_signed"));

    m.def("swath_compute_full_distance_map", &wrap_swath_compute_full_distance_map,
          "Compute full (unclipped) distance map from track",
          py::arg("distance"),
          py::arg("nearest_segment") = py::none(),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("dims"),
          py::arg("cellsize"),
          py::arg("dem") = py::none(),
          py::arg("mask") = py::none(),
          py::arg("compute_signed"));

    m.def("swath_transverse", &wrap_swath_transverse,
          "Compute transverse swath profile (averaged cross-section)",
          py::arg("bin_distances"),
          py::arg("bin_means"),
          py::arg("bin_stddevs"),
          py::arg("bin_mins"),
          py::arg("bin_maxs"),
          py::arg("bin_counts"),
          py::arg("bin_medians") = py::none(),
          py::arg("bin_q1") = py::none(),
          py::arg("bin_q3") = py::none(),
          py::arg("percentile_list") = py::none(),
          py::arg("bin_percentiles") = py::none(),
          py::arg("dem"),
          py::arg("distance_from_track"),
          py::arg("dims"),
          py::arg("half_width"),
          py::arg("bin_resolution"),
          py::arg("normalize") = 0);

    m.def("swath_longitudinal", &wrap_swath_longitudinal,
          "Compute longitudinal swath profile (along-track variation)",
          py::arg("point_means"),
          py::arg("point_stddevs"),
          py::arg("point_mins"),
          py::arg("point_maxs"),
          py::arg("point_counts"),
          py::arg("point_medians") = py::none(),
          py::arg("point_q1") = py::none(),
          py::arg("point_q3") = py::none(),
          py::arg("percentile_list") = py::none(),
          py::arg("point_percentiles") = py::none(),
          py::arg("dem"),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("distance_from_track"),
          py::arg("dims"),
          py::arg("cellsize"),
          py::arg("half_width"),
          py::arg("binning_distance"),
          py::arg("n_points_regression"),
          py::arg("use_segment_seeds"),
          py::arg("skip"),
          py::arg("result_track_i") = py::none(),
          py::arg("result_track_j") = py::none());

    m.def("swath_longitudinal_windowed", &wrap_swath_longitudinal_windowed,
          "Compute windowed longitudinal swath profile",
          py::arg("point_means"),
          py::arg("point_stddevs"),
          py::arg("point_mins"),
          py::arg("point_maxs"),
          py::arg("point_counts"),
          py::arg("point_medians") = py::none(),
          py::arg("point_q1") = py::none(),
          py::arg("point_q3") = py::none(),
          py::arg("percentile_list") = py::none(),
          py::arg("point_percentiles") = py::none(),
          py::arg("dem"),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("dims"),
          py::arg("cellsize"),
          py::arg("half_width"),
          py::arg("binning_distance"),
          py::arg("n_points_regression"),
          py::arg("skip"),
          py::arg("result_track_i") = py::none(),
          py::arg("result_track_j") = py::none());

    m.def("swath_windowed_get_point_samples", &wrap_swath_windowed_get_point_samples,
          "Get pixel coordinates for a point's oriented-rectangle window",
          py::arg("pixels_i"),
          py::arg("pixels_j"),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("point_index"),
          py::arg("dims"),
          py::arg("cellsize"),
          py::arg("half_width"),
          py::arg("binning_distance"),
          py::arg("n_points_regression"));

    m.def("swath_get_point_pixels", &wrap_swath_get_point_pixels,
          "Get pixel coordinates associated with a single track point",
          py::arg("pixels_i"),
          py::arg("pixels_j"),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("point_index"),
          py::arg("distance_from_track"),
          py::arg("dims"),
          py::arg("cellsize"),
          py::arg("half_width"),
          py::arg("binning_distance"),
          py::arg("n_points_regression"),
          py::arg("use_segment_seeds"));

    m.def("sample_points_between_refs", &wrap_sample_points_between_refs,
          "Sample points between reference points using Bresenham-like logic",
          py::arg("out_i"),
          py::arg("out_j"),
          py::arg("ref_i"),
          py::arg("ref_j"),
          py::arg("close_loop"),
          py::arg("use_d4"));

    m.def("simplify_line", &wrap_simplify_line,
          "Simplify a polyline using EF engine",
          py::arg("out_i"),
          py::arg("out_j"),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("tolerance"),
          py::arg("method"));

    m.attr("SIMPLIFY_FIXED_N") = py::int_(SIMPLIFY_FIXED_N);
    m.attr("SIMPLIFY_KNEEDLE") = py::int_(SIMPLIFY_KNEEDLE);
    m.attr("SIMPLIFY_AIC") = py::int_(SIMPLIFY_AIC);
    m.attr("SIMPLIFY_BIC") = py::int_(SIMPLIFY_BIC);
    m.attr("SIMPLIFY_MDL") = py::int_(SIMPLIFY_MDL);
    m.attr("SIMPLIFY_VW_AREA") = py::int_(SIMPLIFY_VW_AREA);
    m.attr("SIMPLIFY_L_METHOD") = py::int_(SIMPLIFY_L_METHOD);
}
