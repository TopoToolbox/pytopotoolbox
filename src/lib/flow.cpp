extern "C" {
  #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

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
                       bitmap2.mutable_data(), weights2.mutable_data(),
                       dims);
}


PYBIND11_MODULE(_flow, m) {
  m.def("edgeset_count", &wrap_edgeset_count);
  m.def("edgeset_scan", &wrap_edgeset_scan);
  m.def("edgeset_count_merged", &wrap_edgeset_count_merged);
  m.def("edgeset_merge", &wrap_edgeset_merge);
}
