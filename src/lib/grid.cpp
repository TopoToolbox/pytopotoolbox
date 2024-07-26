// This file contains bindings for the fillsinks function using PyBind11.

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
//   nrows: Number of rows in the input DEM.
//   ncols: Number of columns in the input DEM.

void wrap_fillsinks(py::array_t<float> output, py::array_t<float> dem, ptrdiff_t nrows, ptrdiff_t ncols){
    float *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    // calling the fillsinks function of libtopotoolbox
    fillsinks(output_ptr, dem_ptr, nrows, ncols);
}

// wrap_identifyflats: 
// Parameters:
//   output: A NumPy array to store the output, where flats, sill amd presills will be marked.
//   dem: A NumPy array representing the digital elevation model.
//   nrows: Number of rows in the input DEM.
//   ncols: Number of columns in the input DEM.

void wrap_identifyflats(py::array_t<int32_t> output, py::array_t<float> dem, ptrdiff_t nrows, ptrdiff_t ncols){
    int32_t *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    identifyflats(output_ptr, dem_ptr, nrows, ncols);
}

void wrap_excesstopography_fsm2d(py::array_t<float> excess, py::array_t<float> dem, py::array_t<float> threshold_slopes, float cellsize, ptrdiff_t nrows, ptrdiff_t ncols){
    float *excess_ptr = excess.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *threshold_slopes_ptr = threshold_slopes.mutable_data();

    excesstopography_fsm2d(excess_ptr, dem_ptr, threshold_slopes_ptr, cellsize, nrows, ncols);
}

void wrap_excesstopography_fmm2d(py::array_t<float> excess, py::array_t<ptrdiff_t> heap, py::array_t<ptrdiff_t> back, py::array_t<float> dem, py::array_t<float> threshold_slopes, float cellsize, ptrdiff_t nrows, ptrdiff_t ncols){
    float *excess_ptr = excess.mutable_data();
    ptrdiff_t *heap_ptr = heap.mutable_data();
    ptrdiff_t *back_ptr = back.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *threshold_slopes_ptr = threshold_slopes.mutable_data();

    excesstopography_fmm2d(excess_ptr, heap_ptr, back_ptr, dem_ptr, threshold_slopes_ptr, cellsize, nrows, ncols);
}

// Make wrap_funcname() function available as grid_funcname() to be used by
// by functions in the pytopotoolbox package

PYBIND11_MODULE(_grid, m) {
    m.def("grid_fillsinks", &wrap_fillsinks);
    m.def("grid_identifyflats", &wrap_identifyflats);
    m.def("grid_excesstopography_fsm2d", &wrap_excesstopography_fsm2d);
    m.def("grid_excesstopography_fmm2d", &wrap_excesstopography_fmm2d);
    //m.def("grid_excesstopography_fmm3d", &wrap_excesstopography_fmm3d);
}
