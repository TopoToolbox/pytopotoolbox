// This file contains bindings for the grid functions

// It is expected, that the #include statemnts can raise errors in your IDE.
// The needed files for those imports are only provided during the build process.

extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdlib> 

namespace py = pybind11;

// Wraps the fillsinks function of the libtopotoolbox for use in Python.
// Is necessary to be able to properly pass NumPy arrays.
// Parameters:
//   output: A NumPy array to store the output.
//   dem: A NumPy array representing the digital elevation model.
//   dims: A tuple containing the number of rows and columns.

void wrap_fillsinks(py::array_t<float> output, py::array_t<float> dem,
                    py::array_t<uint8_t> bc,
                    std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();
    uint8_t *bc_ptr = bc.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    fillsinks(output_ptr, dem_ptr, bc_ptr, dims_ptr);
}

void wrap_fillsinks_hybrid(py::array_t<float> output,
                           py::array_t<ptrdiff_t> queue,
                           py::array_t<float> dem,
                           py::array_t<uint8_t> bc,
                           std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();
    ptrdiff_t *queue_ptr = queue.mutable_data();
    uint8_t *bc_ptr = bc.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    fillsinks_hybrid(output_ptr, queue_ptr, dem_ptr, bc_ptr, dims_ptr);
}

// wrap_identifyflats: 
// Parameters:
//   output: A NumPy array to store the output, where flats, sill amd presills will be marked.
//   dem: A NumPy array representing the digital elevation model.
//   dims: A tuple containing the number of rows and columns.

void wrap_identifyflats(py::array_t<int32_t> output, py::array_t<float> dem, 
                        std::tuple<ptrdiff_t, ptrdiff_t> dims){

    int32_t *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    identifyflats(output_ptr, dem_ptr, dims_ptr);
}

// wrap_excesstopography_fsm2d: 
// Parameters:
//   excess: A NumPy array to store the excess topography values computed by the FSM2D algorithm.
//   dem: A NumPy array representing the digital elevation model.
//   threshold_slopes: A NumPy array representing the slope thresholds for excess topography calculation.
//   cellsize: The size of each cell in the DEM.
//   dims: A tuple containing the number of rows and columns.

void wrap_excesstopography_fsm2d(
        py::array_t<float> excess, py::array_t<float> dem,
        py::array_t<float> threshold_slopes, float cellsize, 
        std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *excess_ptr = excess.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *threshold_slopes_ptr = threshold_slopes.mutable_data();
    
    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    excesstopography_fsm2d(excess_ptr, dem_ptr, threshold_slopes_ptr, cellsize, dims_ptr);
}

// wrap_excesstopography_fmm2d: 
// Parameters:
//   excess: A NumPy array to store the excess topography values computed by the FMM2D algorithm.
//   heap: A NumPy array representing the heap used in the FMM2D algorithm.
//   back: A NumPy array representing the backtracking information used in the FMM2D algorithm.
//   dem: A NumPy array representing the digital elevation model.
//   threshold_slopes: A NumPy array representing the slope thresholds for excess topography calculation.
//   cellsize: The size of each cell in the DEM.
//   dims: A tuple containing the number of rows and columns.

void wrap_excesstopography_fmm2d(
        py::array_t<float> excess, py::array_t<ptrdiff_t> heap,
        py::array_t<ptrdiff_t> back, py::array_t<float> dem,
        py::array_t<float> threshold_slopes, float cellsize, 
        std::tuple<ptrdiff_t, ptrdiff_t> dims){

    float *excess_ptr = excess.mutable_data();
    ptrdiff_t *heap_ptr = heap.mutable_data();
    ptrdiff_t *back_ptr = back.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *threshold_slopes_ptr = threshold_slopes.mutable_data();
    
    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    excesstopography_fmm2d(excess_ptr, heap_ptr, back_ptr, dem_ptr, threshold_slopes_ptr, cellsize, dims_ptr);
}

// wrap_gwdt:
// Parameters:
//   dist: A NumPy array to store the geodesic distance values.
//   prev: A NumPy array to store the indices of previous nodes in the optimal path.
//   costs: A NumPy array representing the cost surface for the GWDT calculation.
//   flats: A NumPy array representing flat regions in the DEM.
//   heap: A NumPy array representing the heap used in the GWDT algorithm.
//   back: A NumPy array representing the backtracking information used in the GWDT algorithm.
//   dims: A tuple containing the number of rows and columns.

void wrap_gwdt(
        py::array_t<float> dist, py::array_t<ptrdiff_t> prev,
        py::array_t<float> costs, py::array_t<int32_t> flats,
        py::array_t<ptrdiff_t> heap, py::array_t<ptrdiff_t> back, 
        std::tuple<ptrdiff_t, ptrdiff_t> dims){
            
    float *dist_ptr = dist.mutable_data();
    ptrdiff_t *prev_ptr = prev.mutable_data();
    float *costs_ptr = costs.mutable_data();
    int32_t *flats_ptr = flats.mutable_data();
    ptrdiff_t *heap_ptr = heap.mutable_data();
    ptrdiff_t *back_ptr = back.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    gwdt(dist_ptr, prev_ptr, costs_ptr, flats_ptr, heap_ptr, back_ptr, dims_ptr);
}

// wrap_gwdt_computecosts:
// Parameters:
//   costs: A NumPy array to store the computed cost values.
//   conncomps: A NumPy array representing connected components in the DEM.
//   flats: A NumPy array representing flat regions in the DEM.
//   original_dem: A NumPy array representing the original digital elevation model (DEM).
//   filled_dem: A NumPy array representing the filled digital elevation model (DEM).
//   dims: A tuple containing the number of rows and columns.

void wrap_gwdt_computecosts(
        py::array_t<float> costs,py::array_t<ptrdiff_t> conncomps,
        py::array_t<int32_t> flats, py::array_t<float> original_dem,
        py::array_t<float>filled_dem, std::tuple<ptrdiff_t, ptrdiff_t>dims){

    float *costs_ptr = costs.mutable_data();
    ptrdiff_t *conncomps_ptr = conncomps.mutable_data();
    int32_t *flats_ptr = flats.mutable_data();
    float *original_dem_ptr = original_dem.mutable_data();
    float *filled_dem_ptr = filled_dem.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    gwdt_computecosts(costs_ptr,conncomps_ptr,flats_ptr,original_dem_ptr,filled_dem_ptr, dims_ptr);
}

// wrap_flow_routing_d8_carve:
// Parameters:
//   node: A NumPy array representing the topologically sorted list of
//         flow network vertices.
//   direction: A NumPy array to store the computed flow directions.
//   dem: A NumPy array representing the digital elevation model (DEM).
//   dist: A NumPy array to store the computed flow distances.
//   flats: A NumPy array representing flat regions in the DEM.
//   dims: A tuple containing the number of rows and columns.

void wrap_flow_routing_d8_carve(
        py::array_t<ptrdiff_t> node, py::array_t<uint8_t> direction, 
        py::array_t<float> dem, py::array_t<float> dist, 
        py::array_t<int32_t> flats, std::tuple<ptrdiff_t, ptrdiff_t> dims){
    
    ptrdiff_t *node_ptr = node.mutable_data();
    uint8_t *direction_ptr = direction.mutable_data();
    float *dem_ptr = dem.mutable_data();
    float *dist_ptr = dist.mutable_data();
    int32_t *flats_ptr = flats.mutable_data();
        
    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();

    flow_routing_d8_carve(node_ptr, direction_ptr, dem_ptr, dist_ptr, flats_ptr, dims_ptr);
}

// wrap_flow_routing_d8_edgelist:
// Parameters:
//   source: A NumPy array representing the source cells for flow routing.
//   target: A NumPy array to store the computed target cells for flow routing.
//   node: A Numpy array containing the topologically sorted list of vertices.
//   direction: A NumPy array representing the computed flow directions.
//   dims: A tuple containing the number of rows and columns.

ptrdiff_t wrap_flow_routing_d8_edgelist(
        py::array_t<ptrdiff_t> source,
        py::array_t<ptrdiff_t> target,
        py::array_t<ptrdiff_t> node,
        py::array_t<uint8_t> direction,
        std::tuple<ptrdiff_t,ptrdiff_t> dims) {

    ptrdiff_t *source_ptr = source.mutable_data();
    ptrdiff_t *target_ptr = target.mutable_data();

    ptrdiff_t *node_ptr = node.mutable_data();
    uint8_t *direction_ptr = direction.mutable_data(); 

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    return flow_routing_d8_edgelist(source_ptr, target_ptr, node_ptr, direction_ptr, dims_ptr);
}

// wrap_gradient8:
// Parameters:
//   output: A NumPy array which will store the resulting gradient of each cell.
//   dem: A NumPy array containing the digital elevation model.
//   cellsize: A float representing the spacial resolution of the dem.
//   use_mp: A int which will decide if OpenMP will be used.
//   dims: A tuple containing the number of rows and columns.

void wrap_gradient8(
        py::array_t<float> output, py::array_t<float> dem, float cellsize, 
        int use_mp, std::tuple<ptrdiff_t,ptrdiff_t> dims){

    float *output_ptr = output.mutable_data();
    float *dem_ptr = dem.mutable_data();

    std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
    ptrdiff_t *dims_ptr = dims_array.data();
    gradient8(output_ptr, dem_ptr, cellsize, use_mp, dims_ptr);
}

void wrap_hillshade(py::array_t<float> output,
                    py::array_t<float>nx, py::array_t<float> ny, py::array_t<float> nz,
                    py::array_t<float> dem,
                    float azimuth, float altitude, float cellsize,
                    std::tuple<ptrdiff_t, ptrdiff_t> dims) {
  
  float *output_ptr = output.mutable_data();
  float *nx_ptr = nx.mutable_data();
  float *ny_ptr = ny.mutable_data();
  float *nz_ptr = nz.mutable_data();
  float *dem_ptr = dem.mutable_data();

  std::array<ptrdiff_t, 2> dims_array = {std::get<0>(dims), std::get<1>(dims)};
  hillshade(output_ptr, nx_ptr, ny_ptr, nz_ptr,
            dem_ptr, azimuth, altitude, cellsize,
            dims_array.data());
}

// Make wrap_funcname() function available as funcname() to be used by
// by functions in the pytopotoolbox package

PYBIND11_MODULE(_grid, m) {
    m.def("fillsinks", &wrap_fillsinks);
    m.def("fillsinks_hybrid", &wrap_fillsinks_hybrid);
    m.def("identifyflats", &wrap_identifyflats);
    m.def("excesstopography_fsm2d", &wrap_excesstopography_fsm2d);
    m.def("excesstopography_fmm2d", &wrap_excesstopography_fmm2d);
    m.def("gwdt", &wrap_gwdt);
    m.def("gwdt_computecosts", &wrap_gwdt_computecosts);
    m.def("flow_routing_d8_carve", &wrap_flow_routing_d8_carve);
    m.def("flow_routing_d8_edgelist", &wrap_flow_routing_d8_edgelist);
    m.def("gradient8", &wrap_gradient8);
    m.def("hillshade", &wrap_hillshade);
}
