extern "C" {
  #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_lib, m) {
  m.def("has_topotoolbox",&has_topotoolbox);
}
