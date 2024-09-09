// This file contains bindings for the flow functions

// It is expected, that the #include statemnts can raise errors in your IDE.
// The needed files for those imports are only provided during the build process.

extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// wrap_flow_accumulation:
// Parameters:
//   acc: A 2D NumPy array (float) representing the accumulation matrix. 
//        This matrix will be modified in-place to store the accumulated flow
//        values based on the given flow directions and weights.
//
//   source: A 2D NumPy array (ptrdiff_t) representing the source cell indices.
//           This array contains the cell indices (or indices) from which the 
//           flow originates or accumulates.
//
//   direction: A 2D NumPy array (uint8_t) representing the flow direction for 
//              each cell in the grid. This array contains encoded flow 
//              direction values (e.g., D8 flow directions) to specify how 
//              flow moves between cells in the grid.
//
//   weights: A 2D NumPy array (float) representing the weights or flow values
//            assigned to each cell. These values are used to weight the
//            accumulation of flow from one cell to its neighbors.
//
//   dims: A tuple values representing the dimensions of the grid (number of 
//         rows and  columns) where flow accumulation is performed.

void wrap_flow_accumulation(
        py::array_t<float> acc, py::array_t<ptrdiff_t> source, 
        py::array_t<uint8_t> direction, py::array_t<float> weights,
        std::tuple<ptrdiff_t,ptrdiff_t> dims){

    float *acc_ptr = acc.mutable_data();
    ptrdiff_t *source_ptr = source.mutable_data();
    uint8_t *direction_ptr = direction.mutable_data();
    float *weights_ptr = weights.mutable_data();
    
    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    flow_accumulation(acc_ptr, source_ptr, direction_ptr, weights_ptr, dims_ptr);
}

// Make wrap_funcname() function available as grid_funcname() to be used by
// by functions in the pytopotoolbox package

PYBIND11_MODULE(_flow, m) {
    m.def("flow_accumulation", &wrap_flow_accumulation);
}
