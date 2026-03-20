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


// Wraps swath_frontier_distance_map
void wrap_swath_frontier_distance_map(
        py::array_t<float> best_abs,
        py::object signed_dist_obj,
        py::object nearest_point_obj,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float max_dist_px,
        py::object mask_obj){

    float *best_abs_ptr = best_abs.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    float *signed_dist_ptr = signed_dist_obj.is_none() ? nullptr : signed_dist_obj.cast<py::array_t<float>>().mutable_data();
    ptrdiff_t *near_pt_ptr = nearest_point_obj.is_none() ? nullptr : nearest_point_obj.cast<py::array_t<ptrdiff_t>>().mutable_data();
    int8_t *mask_ptr = mask_obj.is_none() ? nullptr : mask_obj.cast<py::array_t<int8_t>>().mutable_data();

    swath_frontier_distance_map(best_abs_ptr, signed_dist_ptr, near_pt_ptr,
                                track_i_ptr, track_j_ptr, n_track_points,
                                dims_ptr, max_dist_px, mask_ptr);
}

// Wraps swath_boundary_dijkstra
void wrap_swath_boundary_dijkstra(
        py::array_t<float> dist_out,
        py::array_t<int8_t> swath_mask,
        py::array_t<ptrdiff_t> seeds,
        std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *dist_ptr = dist_out.mutable_data();
    const int8_t *mask_ptr = swath_mask.data();
    const ptrdiff_t *seeds_ptr = seeds.data();
    ptrdiff_t n_seeds = seeds.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    swath_boundary_dijkstra(dist_ptr, mask_ptr, seeds_ptr, n_seeds, dims_ptr);
}

// Wraps voronoi_ridge_to_centreline
ptrdiff_t wrap_voronoi_ridge_to_centreline(
        py::array_t<float> centre_line_i,
        py::array_t<float> centre_line_j,
        py::array_t<float> centre_width,
        py::array_t<float> dist_pos,
        py::array_t<float> dist_neg,
        py::array_t<float> best_abs,
        float hw_px,
        py::array_t<ptrdiff_t> nearest_point,
        py::array_t<float> track_i,
        py::array_t<float> track_j,
        std::tuple<ptrdiff_t, ptrdiff_t> dims,
        float cellsize){

    float *cli_ptr = centre_line_i.mutable_data();
    float *clj_ptr = centre_line_j.mutable_data();
    float *cw_ptr = centre_width.mutable_data();
    const float *dp_ptr = dist_pos.data();
    const float *dn_ptr = dist_neg.data();
    const float *ba_ptr = best_abs.data();
    const ptrdiff_t *npt_ptr = nearest_point.data();
    const float *ti_ptr = track_i.data();
    const float *tj_ptr = track_j.data();
    ptrdiff_t n_track_points = track_i.size();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    return voronoi_ridge_to_centreline(cli_ptr, clj_ptr, cw_ptr, dp_ptr, dn_ptr,
                                       ba_ptr, hw_px, npt_ptr, ti_ptr, tj_ptr,
                                       n_track_points, dims_ptr, cellsize);
}

// Wraps thin_rasterised_line_to_D8
ptrdiff_t wrap_thin_rasterised_line_to_D8(
        py::array_t<float> centre_line_i,
        py::array_t<float> centre_line_j,
        py::array_t<float> centre_width,
        ptrdiff_t n_centre,
        std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *cli_ptr = centre_line_i.mutable_data();
    float *clj_ptr = centre_line_j.mutable_data();
    float *cw_ptr = centre_width.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    return thin_rasterised_line_to_D8(cli_ptr, clj_ptr, cw_ptr, n_centre, dims_ptr);
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
        py::object nearest_point_obj,
        py::array_t<float> cum_dist,
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

    ptrdiff_t *near_pt_ptr = nearest_point_obj.is_none() ? nullptr : nearest_point_obj.cast<py::array_t<ptrdiff_t>>().mutable_data();
    float *cum_dist_ptr = cum_dist.mutable_data();
    float *res_i_ptr = result_track_i_obj.is_none() ? nullptr : result_track_i_obj.cast<py::array_t<float>>().mutable_data();
    float *res_j_ptr = result_track_j_obj.is_none() ? nullptr : result_track_j_obj.cast<py::array_t<float>>().mutable_data();

    return swath_longitudinal(means_ptr, std_ptr, min_ptr, max_ptr, counts_ptr,
                               medians_ptr, q1_ptr, q3_ptr, perc_list_ptr, n_percs, point_percs_ptr,
                               dem_ptr, track_i_ptr, track_j_ptr, n_track_points,
                               dist_map_ptr, dims_ptr, cellsize, half_width, binning_distance,
                               near_pt_ptr, cum_dist_ptr, skip, res_i_ptr, res_j_ptr);
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
        py::object nearest_point_obj,
        py::array_t<float> cum_dist){

    ptrdiff_t *pi_ptr = pixels_i.mutable_data();
    ptrdiff_t *pj_ptr = pixels_j.mutable_data();
    float *track_i_ptr = track_i.mutable_data();
    float *track_j_ptr = track_j.mutable_data();
    ptrdiff_t n_track_points = track_i.size();
    float *dist_map_ptr = distance_from_track.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    ptrdiff_t *near_pt_ptr = nearest_point_obj.is_none() ? nullptr : nearest_point_obj.cast<py::array_t<ptrdiff_t>>().mutable_data();
    float *cum_dist_ptr = cum_dist.mutable_data();

    return swath_get_point_pixels(pi_ptr, pj_ptr, track_i_ptr, track_j_ptr, n_track_points,
                                   point_index, dist_map_ptr, dims_ptr, cellsize, half_width,
                                   binning_distance, near_pt_ptr, cum_dist_ptr);
}

// wrap_rasterize_path
ptrdiff_t wrap_rasterize_path(
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

    return rasterize_path(oi_ptr, oj_ptr, ri_ptr, rj_ptr, n_refs, close_loop, use_d4);
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


    m.def("swath_frontier_distance_map", &wrap_swath_frontier_distance_map,
          "Frontier Dijkstra distance map, raw pixel-unit outputs",
          py::arg("best_abs"),
          py::arg("signed_dist") = py::none(),
          py::arg("nearest_point") = py::none(),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("dims"),
          py::arg("max_dist_px"),
          py::arg("mask") = py::none());

    m.def("swath_boundary_dijkstra", &wrap_swath_boundary_dijkstra,
          "Inward D8 Dijkstra from boundary seed pixels",
          py::arg("dist_out"),
          py::arg("swath_mask"),
          py::arg("seeds"),
          py::arg("dims"));

    m.def("voronoi_ridge_to_centreline", &wrap_voronoi_ridge_to_centreline,
          "Extract Voronoi ridge pixels between two boundary wavefronts",
          py::arg("centre_line_i"),
          py::arg("centre_line_j"),
          py::arg("centre_width"),
          py::arg("dist_pos"),
          py::arg("dist_neg"),
          py::arg("best_abs"),
          py::arg("hw_px"),
          py::arg("nearest_point"),
          py::arg("track_i"),
          py::arg("track_j"),
          py::arg("dims"),
          py::arg("cellsize"));

    m.def("thin_rasterised_line_to_D8", &wrap_thin_rasterised_line_to_D8,
          "Thin a rasterised polyline by removing staircase elbows",
          py::arg("centre_line_i"),
          py::arg("centre_line_j"),
          py::arg("centre_width"),
          py::arg("n_centre"),
          py::arg("dims"));

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
          py::arg("nearest_point") = py::none(),
          py::arg("cum_dist"),
          py::arg("skip"),
          py::arg("result_track_i") = py::none(),
          py::arg("result_track_j") = py::none());


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
          py::arg("nearest_point") = py::none(),
          py::arg("cum_dist"));

    m.def("rasterize_path", &wrap_rasterize_path,
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

}
