cmake_minimum_required(VERSION 3.15)

project(${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# include libtopotoolbox
include(FetchContent)
FetchContent_Declare(
  topotoolbox
  GIT_REPOSITORY https://github.com/TopoToolbox/libtopotoolbox.git
  GIT_TAG ebe13075246a2a3fd0d42111703bda4cc5c3a5d7
)
FetchContent_MakeAvailable(topotoolbox)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_grid src/lib/grid.cpp)
target_link_libraries(_grid PRIVATE topotoolbox)

pybind11_add_module(_flow src/lib/flow.cpp)
target_link_libraries(_flow PRIVATE topotoolbox)

pybind11_add_module(_stream src/lib/stream.cpp)
target_link_libraries(_stream PRIVATE topotoolbox)

pybind11_add_module(_graphflood src/lib/graphflood.cpp)
target_link_libraries(_graphflood PRIVATE topotoolbox)

pybind11_add_module(_morphology src/lib/morphology.cpp)
target_link_libraries(_morphology PRIVATE topotoolbox)

install(TARGETS _grid _flow _stream _graphflood _morphology LIBRARY DESTINATION topotoolbox)
