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

void wrap_traverse_up_u32_and(py::array_t<uint32_t> output,
                              py::array_t<uint32_t> input,
                              py::array_t<ptrdiff_t> source,
                              py::array_t<ptrdiff_t> target) {

  uint32_t *output_ptr = output.mutable_data();
  uint32_t *input_ptr = input.mutable_data();
  ptrdiff_t *source_ptr = source.mutable_data();
  ptrdiff_t *target_ptr = target.mutable_data();

  ptrdiff_t edge_count = source.size(); 

  traverse_up_u32_and(output_ptr, input_ptr, source_ptr,
                      target_ptr, edge_count);  
}

void wrap_traverse_down_f32_max_add(py::array_t<float> output,
                                    py::array_t<float> input,
                                    py::array_t<ptrdiff_t> source,
                                    py::array_t<ptrdiff_t> target) {

  float *output_ptr = output.mutable_data();
  float *input_ptr = input.mutable_data();
  ptrdiff_t *source_ptr = source.mutable_data();
  ptrdiff_t *target_ptr = target.mutable_data();

  ptrdiff_t edge_count = source.size();

  traverse_down_f32_max_add(output_ptr, input_ptr, source_ptr,
                            target_ptr, edge_count);
}

void wrap_traverse_down_f32_add_mul(py::array_t<float> output,
                                    py::array_t<float> input,
                                    py::array_t<ptrdiff_t> source,
                                    py::array_t<ptrdiff_t> target) {
  float *output_ptr = output.mutable_data();
  float *input_ptr = input.mutable_data();
  ptrdiff_t *source_ptr = source.mutable_data();
  ptrdiff_t *target_ptr = target.mutable_data();

  ptrdiff_t edge_count = source.size();

  traverse_down_f32_add_mul(output_ptr, input_ptr, source_ptr,
                            target_ptr, edge_count);
}

void wrap_edgelist_degree(py::array_t<uint8_t> indegree,
                          py::array_t<uint8_t> outdegree,
                          py::array_t<ptrdiff_t> source,
                          py::array_t<ptrdiff_t> target) {
  uint8_t *indegree_ptr = indegree.mutable_data();
  uint8_t *outdegree_ptr = outdegree.mutable_data();
  ptrdiff_t *source_ptr = source.mutable_data();
  ptrdiff_t *target_ptr = target.mutable_data();

  ptrdiff_t node_count = indegree.size();
  ptrdiff_t edge_count = source.size();

  edgelist_degree(indegree_ptr, outdegree_ptr, source_ptr,
                  target_ptr, node_count, edge_count);
}

void wrap_propagatevaluesupstream_u8(py::array_t<uint8_t> data,
                                     py::array_t<ptrdiff_t> source,
                                     py::array_t<ptrdiff_t> target) {

  propagatevaluesupstream_u8(data.mutable_data(),
                             source.mutable_data(),
                             target.mutable_data(),
                             source.size());
}

PYBIND11_MODULE(_stream, m) {
  m.def("streamquad_trapz_f32", &wrap_streamquad_trapz_f32);
  m.def("streamquad_trapz_f64", &wrap_streamquad_trapz_f64);
  m.def("traverse_up_u32_and", &wrap_traverse_up_u32_and);
  m.def("traverse_down_f32_max_add", &wrap_traverse_down_f32_max_add);
  m.def("traverse_down_f32_add_mul", &wrap_traverse_down_f32_add_mul);
  m.def("edgelist_degree", &wrap_edgelist_degree);
  m.def("propagatevaluesupstream_u8", &wrap_propagatevaluesupstream_u8);
}
