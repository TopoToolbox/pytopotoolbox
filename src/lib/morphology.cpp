extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib> 

namespace py = pybind11;

// wrap_reconstruct:
// Parameters:
//   marker: A NumPy array containing the marker array.
//   dem: A NumPy array containing the digital elevation model.
//   dims: A tuple containing the number of rows and columns.

void wrap_reconstruct(py::array_t<float> marker, py::array_t<float> dem, 
    std::tuple<ptrdiff_t,ptrdiff_t> dims){
    float *marker_ptr = marker.mutable_data();
    float *dem_ptr = dem.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    reconstruct(marker_ptr, dem_ptr, dims_ptr);
}

PYBIND11_MODULE(_morphology, m) {
    m.def("reconstruct", &wrap_reconstruct);
}