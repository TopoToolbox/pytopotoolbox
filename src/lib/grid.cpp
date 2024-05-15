extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void wrap_fillsinks(py::array_t<float> output, py::array_t<float> dem, ptrdiff_t nrows, ptrdiff_t ncols){
    float *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    fillsinks(output_ptr, dem_ptr, nrows, ncols);
}

void wrap_identifyflats(py::array_t<int32_t> output, py::array_t<float> dem, ptrdiff_t nrows, ptrdiff_t ncols){
    int32_t *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    identifyflats(output_ptr, dem_ptr, nrows, ncols);
}

PYBIND11_MODULE(_grid, m) {
    m.def("grid_fillsinks", &wrap_fillsinks);
    m.def("grid_identifyflats", &wrap_identifyflats);
}
