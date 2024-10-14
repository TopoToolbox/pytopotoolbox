// This file contains bindings for the fillsinks function using PyBind11.

extern "C" {
    #include <topotoolbox.h>
}

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdbool.h>
#include <stdio.h>
#include <iostream>

namespace py = pybind11;


/*
Runs the full graphflood's algorithm as described in Gailleton et al., 2024

Z:              1D numpy array (n nodes) of topography (type np.float32)
hw:             1D numpy array (n nodes) of flow depth type np.float32)
BCs:            1D numpy array (n nodes) of boundary codes (type np.uint8)
Precipitations: 1D numpy array (n nodes) of precipitation rates in m.s-1 (type np.float32)
manning:        1D numpy array (n nodes) of friction coefficient (type np.float32)
dim:            [nrows,ncolumns] for row major (e.g. python) or [ncolumns, nrows] for column major. Numpy array as np.uint64.
dt:             time step (s ~ although this is not simulated time as we make the steady low assumption)
dx:             spatial step (m)
D8:             true to include diagonal paths
N_iterations:   how many iterations to run

B.G. (last modification: 08/2024)
*/
void wrap_graphflood_full(
    py::array_t<GF_FLOAT> Z,
    py::array_t<GF_FLOAT> hw,
    py::array_t<uint8_t>  BCs,
    py::array_t<GF_FLOAT> Precipitations,
    py::array_t<GF_FLOAT> manning,
    py::array_t<GF_UINT>  dim,
    GF_FLOAT              dt,
    GF_FLOAT              dx,
    bool                  SFD,
    bool                  D8,
    GF_UINT               N_iterations,
    GF_FLOAT              step
    ){

    // numpy arrays to pointers
    GF_FLOAT* Z_ptr              = Z.mutable_data()              ;
    GF_FLOAT* hw_ptr             = hw.mutable_data()             ;
    uint8_t*  BCs_ptr            = BCs.mutable_data()            ;
    GF_FLOAT* Precipitations_ptr = Precipitations.mutable_data() ;
    GF_FLOAT* manning_ptr        = manning.mutable_data()        ;
    GF_UINT*  dim_ptr            = dim.mutable_data()            ;

    // calling the C function
    graphflood_full(Z_ptr, hw_ptr, BCs_ptr, Precipitations_ptr, manning_ptr, dim_ptr, dt, dx, SFD, D8, N_iterations, step);
}


/*
Computes a single flow graph compatible with the boundary conditions system of graphflood

topo: 1D numpy array (n nodes) of topography (type np.float32)
Sreceivers: 1D numpy array (n nodes), flat index of steepest receiver (type np.uint64)
distToReceivers: 1D numpy array (n nodes), distance to steepest receiver (type np.float32)
Sdonors: 1D numpy array (n nodes * 4 or 8), flat index of donors (type np.uint64)
NSdonors: 1D numpy array (n nodes ), flat index of number of steepest donors (type np.uint8)
Stack: Topologically ordered nodes from downstream to upstream (stakc in Braun and Willett 2013)
BCs: 1D numpy array (n nodes) of boundary codes (type np.uint8)
dim: [nrows,ncolumns] for row major (e.g. python) or [ncolumns, nrows] for column major. Numpy array as np.uint64.
dx: spatial step (m)
D8: true to include diagonal paths
PF: true to also fill the topography using priority flood (Barnes, 2014)

B.G. (last modification: 08/2024)
*/
void wrap_compute_sfgraph(
    py::array_t<GF_FLOAT>    topo, 
    py::array_t<GF_UINT>     Sreceivers, 
    py::array_t<GF_FLOAT>    distToReceivers, 
    py::array_t<GF_UINT>     Sdonors, 
    py::array_t<uint8_t>     NSdonors,
    py::array_t<GF_UINT>     Stack,
    py::array_t<uint8_t>     BCs,
    py::array_t<GF_UINT>     dim,
    GF_FLOAT                 dx,
    bool                     D8,
    bool                     PF,
    GF_FLOAT                 step
    ){

    GF_FLOAT *topo_ptr             =   topo.mutable_data()            ;
    GF_UINT  *Sreceivers_ptr       =   Sreceivers.mutable_data()      ;
    GF_FLOAT *distToReceivers_ptr  =   distToReceivers.mutable_data() ;
    GF_UINT  *Sdonors_ptr          =   Sdonors.mutable_data()         ;
    uint8_t  *NSdonors_ptr         =   NSdonors.mutable_data()        ;
    GF_UINT  *Stack_ptr            =   Stack.mutable_data()           ;
    uint8_t  *BCs_ptr              =   BCs.mutable_data()             ;
    GF_UINT  *dim_ptr              =   dim.mutable_data()             ;
    
    if(PF)
        compute_sfgraph_priority_flood(topo_ptr, Sreceivers_ptr, distToReceivers_ptr, Sdonors_ptr, NSdonors_ptr, Stack_ptr, BCs_ptr, dim_ptr,  dx,  D8, step);
    else
        compute_sfgraph(topo_ptr, Sreceivers_ptr, distToReceivers_ptr, Sdonors_ptr, NSdonors_ptr, Stack_ptr, BCs_ptr, dim_ptr,  dx,  D8);

}

void wrap_compute_drainage_area_single_flow(
    py::array_t<GF_FLOAT> output,
    py::array_t<GF_UINT> Sreceivers,
    py::array_t<GF_UINT> Stack,
    py::array_t<GF_UINT> dim,
    GF_FLOAT dx
    ){

    GF_FLOAT* output_ptr = output.mutable_data();
    GF_UINT* Sreceivers_ptr = Sreceivers.mutable_data();
    GF_UINT* Stack_ptr = Stack.mutable_data();
    GF_UINT* dim_ptr = dim.mutable_data();


    compute_drainage_area_single_flow(output_ptr, Sreceivers_ptr, Stack_ptr, dim_ptr, dx);
}

void wrap_compute_priority_flood_plus_stack(
	py::array_t<GF_FLOAT>    topo, 
	py::array_t<GF_UINT>     Stack,
	py::array_t<uint8_t>     BCs,
	py::array_t<GF_UINT>     dim,
	bool D8 ,
    GF_FLOAT step
	){
	
    GF_FLOAT* topo_ptr = topo.mutable_data();
    GF_UINT* Stack_ptr = Stack.mutable_data();
    uint8_t* BCs_ptr = BCs.mutable_data();
	GF_UINT* dim_ptr = dim.mutable_data();
	
	// First priority flooding and calculating stack
    compute_priority_flood_plus_topological_ordering(topo_ptr, Stack_ptr, BCs_ptr, dim_ptr, D8,step);
}

void wrap_compute_priority_flood(
    py::array_t<GF_FLOAT>    topo, 
    py::array_t<uint8_t>     BCs,
    py::array_t<GF_UINT>     dim,
    bool D8 ,
    GF_FLOAT step
    ){
    
    GF_FLOAT* topo_ptr = topo.mutable_data();
    uint8_t* BCs_ptr = BCs.mutable_data();
    GF_UINT* dim_ptr = dim.mutable_data();
    
    // First priority flooding and calculating stack
    compute_priority_flood(topo_ptr, BCs_ptr, dim_ptr, D8,step);
}


// Make wrap_funcname() function available as grid_funcname() to be used by
// by functions in the pytopotoolbox package

PYBIND11_MODULE(_graphflood, m) {
    m.def("graphflood_run_full", &wrap_graphflood_full);
    m.def("graphflood_sfgraph", &wrap_compute_sfgraph);
    m.def("compute_priority_flood_plus_topological_ordering", &wrap_compute_priority_flood_plus_stack);
    m.def("compute_priority_flood", &wrap_compute_priority_flood);
    m.def("compute_drainage_area_single_flow", &wrap_compute_drainage_area_single_flow);
    

}
