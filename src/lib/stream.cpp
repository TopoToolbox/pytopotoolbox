// This file contains bindings for the stream functions

// It is expected, that the #include statemnts can raise errors in your IDE.
// The needed files for those imports are only provided during the build process.

extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Make wrap_funcname() function available as grid_funcname() to be used by
// by functions in the pytopotoolbox package

PYBIND11_MODULE(_stream, m) {

}