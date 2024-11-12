// This file contains bindings for the stream functions

// It is expected, that the #include statemnts can raise errors in your IDE.
// The needed files for those imports are only provided during the build
// process.

extern "C" {
#include <topotoolbox.h>
}

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Make wrap_funcname() function available as stream_funcname() to be used by
// by functions in the pytopotoolbox package

void wrap_streamquad_trapz_f32(py::array_t<float> integral,
                               py::array_t<float> integrand,
                               py::array_t<ptrdiff_t> source,
                               py::array_t<ptrdiff_t> target,
                               py::array_t<float> weight) {

  float *integral_ptr = integral.mutable_data();
  float *integrand_ptr = integrand.mutable_data();
  ptrdiff_t *source_ptr = source.mutable_data();
  ptrdiff_t *target_ptr = target.mutable_data();
  float *weight_ptr = weight.mutable_data();

  ptrdiff_t edge_count = source.size();

  streamquad_trapz_f32(integral_ptr, integrand_ptr, source_ptr, target_ptr,
                       weight_ptr, edge_count);
}

void wrap_streamquad_trapz_f64(py::array_t<double> integral,
                               py::array_t<double> integrand,
                               py::array_t<ptrdiff_t> source,
                               py::array_t<ptrdiff_t> target,
                               py::array_t<float> weight) {
  double *integral_ptr = integral.mutable_data();
  double *integrand_ptr = integrand.mutable_data();
  ptrdiff_t *source_ptr = source.mutable_data();
  ptrdiff_t *target_ptr = target.mutable_data();
  float *weight_ptr = weight.mutable_data();

  ptrdiff_t edge_count = source.size();

  streamquad_trapz_f64(integral_ptr, integrand_ptr, source_ptr, target_ptr,
                       weight_ptr, edge_count);
}

PYBIND11_MODULE(_stream, m) {
  m.def("streamquad_trapz_f32", &wrap_streamquad_trapz_f32);
  m.def("streamquad_trapz_f64", &wrap_streamquad_trapz_f64);
}
