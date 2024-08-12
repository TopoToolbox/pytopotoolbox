// This file contains bindings for the fillsinks function using PyBind11.

// It is expected, that the #include statemnts can raise errors in your IDE.
// The needed files for those imports are only provided during the build process.

extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Wraps the fillsinks function of the libtopotoolbox for use in Python.
// Is necessary to be able to properly pass NumPy arrays.
// Parameters:
//   output: A NumPy array to store the output.
//   dem: A NumPy array representing the digital elevation model.
// TODO: add arguments

// void wrap_fillsinks(py::array_t<float> output, py::array_t<float> dem, ptrdiff_t nrows, ptrdiff_t ncols){
void wrap_fillsinks(py::array_t<float> output, py::array_t<float> dem, std::tuple<ptrdiff_t, ptrdiff_t> dims){
    float *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    fillsinks(output_ptr, dem_ptr, dims_ptr);
}

// wrap_identifyflats: 
// Parameters:
//   output: A NumPy array to store the output, where flats, sill amd presills will be marked.
//   dem: A NumPy array representing the digital elevation model.
// TODO: add arguments

void wrap_identifyflats(py::array_t<int32_t> output, py::array_t<float> dem, std::tuple<ptrdiff_t, ptrdiff_t> dims){
    int32_t *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    identifyflats(output_ptr, dem_ptr, dims_ptr);
}

// wrap_excesstopography_fsm2d: 
// Parameters:
//   excess: A NumPy array to store the excess topography values computed by the FSM2D algorithm.
//   dem: A NumPy array representing the digital elevation model.
//   threshold_slopes: A NumPy array representing the slope thresholds for excess topography calculation.
//   cellsize: The size of each cell in the DEM.
// TODO: add arguments

void wrap_excesstopography_fsm2d(py::array_t<float> excess, py::array_t<float> dem, py::array_t<float> threshold_slopes, float cellsize, std::tuple<ptrdiff_t, ptrdiff_t> dims){
    float *excess_ptr = excess.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *threshold_slopes_ptr = threshold_slopes.mutable_data();
    
    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    excesstopography_fsm2d(excess_ptr, dem_ptr, threshold_slopes_ptr, cellsize, dims_ptr);
}

// wrap_excesstopography_fmm2d: 
// Parameters:
//   excess: A NumPy array to store the excess topography values computed by the FMM2D algorithm.
//   heap: A NumPy array representing the heap used in the FMM2D algorithm.
//   back: A NumPy array representing the backtracking information used in the FMM2D algorithm.
//   dem: A NumPy array representing the digital elevation model.
//   threshold_slopes: A NumPy array representing the slope thresholds for excess topography calculation.
//   cellsize: The size of each cell in the DEM.
// TODO: add arguments

void wrap_excesstopography_fmm2d(py::array_t<float> excess, py::array_t<ptrdiff_t> heap, py::array_t<ptrdiff_t> back, py::array_t<float> dem, py::array_t<float> threshold_slopes, float cellsize, std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *excess_ptr = excess.mutable_data();
    ptrdiff_t *heap_ptr = heap.mutable_data();
    ptrdiff_t *back_ptr = back.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *threshold_slopes_ptr = threshold_slopes.mutable_data();
    
    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    excesstopography_fmm2d(excess_ptr, heap_ptr, back_ptr, dem_ptr, threshold_slopes_ptr, cellsize, dims_ptr);
}

// wrap_gwdt:
// Parameters:
// TODO: add arguments

void wrap_gwdt(py::array_t<float> dist, py::array_t<ptrdiff_t> prev, py::array_t<float> costs, py::array_t<int32_t> flats, py::array_t<ptrdiff_t> heap, py::array_t<ptrdiff_t> back, std::tuple<ptrdiff_t, ptrdiff_t> dims){
    float *dist_ptr = dist.mutable_data();
    ptrdiff_t *prev_ptr = prev.mutable_data();
    float *costs_ptr = costs.mutable_data();
    int32_t *flats_ptr = flats.mutable_data();
    ptrdiff_t *heap_ptr = heap.mutable_data();
    ptrdiff_t *back_ptr = back.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    gwdt(dist_ptr, prev_ptr, costs_ptr, flats_ptr, heap_ptr, back_ptr, dims_ptr);
}

// Make wrap_funcname() function available as grid_funcname() to be used by
// by functions in the pytopotoolbox package

PYBIND11_MODULE(_grid, m) {
    m.def("grid_fillsinks", &wrap_fillsinks);
    m.def("grid_identifyflats", &wrap_identifyflats);
    m.def("grid_excesstopography_fsm2d", &wrap_excesstopography_fsm2d);
    m.def("grid_excesstopography_fmm2d", &wrap_excesstopography_fmm2d);
    m.def("grid_gwdt", &wrap_gwdt);
}
