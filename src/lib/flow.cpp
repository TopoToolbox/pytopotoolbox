extern "C" {
#include <topotoolbox.h>
}

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

ptrdiff_t wrap_edgeset_count(py::array_t<uint8_t> bitmap) {
  ptrdiff_t dims[2] = {bitmap.shape(0), bitmap.shape(1)};
  return edgeset_count(bitmap.mutable_data(), dims);
}

ptrdiff_t wrap_edgeset_scan(py::array_t<ptrdiff_t> scan,
                            py::array_t<uint8_t> bitmap) {
  ptrdiff_t dims[2] = {bitmap.shape(0), bitmap.shape(1)};
  return edgeset_scan(scan.mutable_data(), bitmap.mutable_data(), dims);
}

ptrdiff_t wrap_edgeset_count_merged(py::array_t<uint8_t> bitmap1,
                                    py::array_t<uint8_t> bitmap2) {
  ptrdiff_t dims[2] = {bitmap1.shape(0), bitmap1.shape(1)};
  return edgeset_count_merged(bitmap1.mutable_data(), bitmap2.mutable_data(),
                              dims);
}

ptrdiff_t wrap_edgeset_merge(py::array_t<float> outweights,
                             py::array_t<ptrdiff_t> outscan,
                             py::array_t<uint8_t> bitmap1,
                             py::array_t<float> weights1,
                             py::array_t<uint8_t> bitmap2,
                             py::array_t<float> weights2) {
  ptrdiff_t dims[2] = {bitmap1.shape(0), bitmap1.shape(1)};
  return edgeset_merge(outweights.mutable_data(), outscan.mutable_data(),
                       bitmap1.mutable_data(), weights1.mutable_data(),
                       bitmap2.mutable_data(), weights2.mutable_data(), dims);
}

void wrap_flow_routing_tsort(
    py::array_t<ptrdiff_t> stream, py::array_t<ptrdiff_t> source,
    py::array_t<ptrdiff_t> target, py::array_t<float> sorted_weight,
    py::array_t<ptrdiff_t> stack, py::array_t<uint8_t> stackdir,
    py::array_t<uint8_t> direction, py::array_t<float> weight,
    py::array_t<ptrdiff_t> weightscan, py::array_t<uint8_t> visited) {

  ptrdiff_t edge_count = weight.size();

  int order = direction.flags() & py::array::c_style ? 1 : 0;

  ptrdiff_t dims[2] = {weightscan.shape(0), weightscan.shape(1)};
  // Reverse the dimensions if in row-major order
  if (order) {
    dims[0] = weightscan.shape(1);
    dims[1] = weightscan.shape(0);
  }


  flow_routing_tsort(stream.mutable_data(), source.mutable_data(),
                     target.mutable_data(), sorted_weight.mutable_data(),
                     stack.mutable_data(), stackdir.mutable_data(),
                     direction.mutable_data(), weight.mutable_data(),
                     weightscan.mutable_data(), visited.mutable_data(),
                     edge_count, dims, order);
}

PYBIND11_MODULE(_flow, m) {
  m.def("edgeset_count", &wrap_edgeset_count);
  m.def("edgeset_scan", &wrap_edgeset_scan);
  m.def("edgeset_count_merged", &wrap_edgeset_count_merged);
  m.def("edgeset_merge", &wrap_edgeset_merge);
  m.def("flow_routing_tsort", &wrap_flow_routing_tsort);
}
