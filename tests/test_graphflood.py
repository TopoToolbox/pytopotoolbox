'''
Set of tests for graphflood
'''

import numpy as np
import matplotlib.pyplot as plt # TO REMOVE I USE FOR CHECKS
import topotoolbox as ttb
import pytest
from rasterio import Affine
from rasterio.coords import BoundingBox


def debug_array_info(array, name):
	print(f"{name} - ID: {id(array)}, Shape: {array.shape}, Contiguous: {array.flags['C_CONTIGUOUS'] or array.flags['F_CONTIGUOUS']}")




def _generate_south_slopping_GridObj(nrows, ncolumns, cellsize, s0, dtype:'float | np.float32 | np.float64' = np.float32):
	'''
	Generates a grid of South Slopping topography ready for graphflood.
	N/E/W are walls and South lets flow escape.
	REturns the bc array and the topo array

	Parameters
	----------
	nrows: Number of rows
	ncolumns: Number of columns
	cellsize: spatial step in m
	s0: slope
	dtype: the numpy type for the object ()


	Returns
	-------
	GridObject
		A grid object with the computed topography.
	Numpy 2D array
		A numpy array of gridobj shape with graphflood-ready boundary conditions.

	'''

	# Create a 2D array of zeros with the specified number of rows and columns
	Z = np.zeros((nrows, ncolumns), dtype = dtype)

	# Calculate the total length of the grid in the north-south direction
	total_length = nrows * cellsize

	# Create an array representing the distance from the southern boundary for each row
	distances = np.arange(nrows, dtype = np.float32) * cellsize

	# Broadcast it in 2D
	distances = np.tile(distances[:, np.newaxis], ncolumns)

	# Calculate the elevation values for each row
	Z = s0 * (total_length - distances)

	# add walls
	Z[0,:] = 9999
	Z[:,[0,-1]] = 9999

	# bcs array
	bcs = np.ones_like(Z,np.uint8) # data inside
	bcs[Z == 9999] = 0 # no data on walls
	bcs[0,:] = 0 # can out in the South
	bcs[-1,:] = 3 # can out in the South

	# Create the GridObject
	grid = ttb.GridObject()
	#
	grid.path = ''
	# name of DEM
	grid.name = 'test'

	# raster metadata
	grid.z = Z.astype(np.float32)
	grid.cellsize = cellsize  # in meters if crs.is_projected == True


	# georeference
	grid.bounds = BoundingBox(0., nrows*cellsize, ncolumns*cellsize, 0.)
	grid.transform = Affine.identity()
	grid.crs = None

	return grid, bcs



def test_analytical_solution():
	'''
	tests an analytical solution of graphflood on a small rectangular channel
	'''
	# Dimensions
	nrows, ncolumns = 126, 20 
	cellsize = 2.
	s0 = 1e-2
	dt = 1e-3

	# param hydro
	qvolin_total = 15
	manning = 0.033

	# Sloping grid
	grid, bcs = _generate_south_slopping_GridObj(nrows, ncolumns, cellsize, s0)
	# Precipitation grid
	p = np.zeros_like(grid.z)
	qvolin_unit = qvolin_total/(ncolumns-2)
	p[1,1:-1] = qvolin_unit/cellsize**2
	# Analytical solution to manning
	hwstar = (manning * qvolin_unit/(cellsize*(s0**0.5)))**(3./5.)
	
	# Simulation
	hw = ttb.run_graphflood(grid, bcs=bcs, dt=dt, p=p, manning=manning, n_iterations=5000)['hw']
	
	# Checking if it works within an error margin
	# Note that I cherry picked these value to be representative of the right solution while minimising the iterations
	assert abs(np.mean(hw[1:50,1:-1]) - hwstar) < 0.02

