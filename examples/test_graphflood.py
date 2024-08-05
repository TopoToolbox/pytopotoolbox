'''
TODO

B.G. 08/2024
'''

import numpy as np
import matplotlib.pyplot as plt
import topotoolbox as ttb
import math as m
import pickle


comp = pickle.load(open("/home/bgailleton/Desktop/code/FastFlood2.0/FastFlood2_Boris/graphflood/paper_scripts/Applications/flood extent/results/GR_P100.pickle", "rb"))
# print(comp['final_hw'])
# quit()
# Flow topology: D8 or D4
D8 = True

# Generating random topography
# dem = ttb.load_dem('tibet')
dem = ttb.read_tif('/home/bgailleton/Desktop/data/green_river_1.tif')
ny = dem.rows
nx = dem.columns

Z = dem.z.ravel(order = 'C')




hw = np.zeros_like(Z)
manning = np.zeros_like(Z) + 0.033
Precipitations = np.zeros_like(Z) + 100 * 1e-3/3600
BCs = np.ones((dem.rows, dem.columns), dtype = np.uint8)
BCs[[0,-1],:] = 3
BCs[:, [0,-1]] = 3
BCs = BCs.ravel(order = 'C')
dim = np.array([ny,nx], dtype = np.uint64)

# print('gulg')
# stack = np.array(ny*nx, dtype = np.uint64)
# ttb.compute_priority_flood_plus_topological_ordering(Z, stack, BCs, dim, D8)
# print('gulgo')
# quit()

dt = 2e-3
dx = dem.cellsize

print(dem.rows * dem.columns)
# quit()

# fig,ax = plt.subplots()
# im = ax.imshow()


fig,ax = plt.subplots()
im = ax.imshow(hw.reshape(dem.rows, dem.columns, order = 'C'), cmap = 'Blues', vmin = 0, vmax = 0.2)
# ax.imshow(Z.reshape(dem.rows, dem.columns, order = 'C'), alpha = 0.)
plt.colorbar(im)
fig.show()

for i in range(100):
	print(i)
	ttb.graphflood_run_full(Z, hw, BCs, Precipitations, manning, dim, dt, dx, False, D8, 10)
	im.set_data(hw.reshape(dem.rows, dem.columns, order = 'C'))

	fig.canvas.draw_idle()
	fig.canvas.start_event_loop(0.001)






# Event handler function
def onclick(event):
	if event.inaxes == ax:
		x = round(event.xdata)
		y = round(event.ydata)
		pixel_value = Z.reshape(dem.rows, dem.columns, order = 'C')[y,x]
		print(f'Pixel at ({x}, {y}) has value: {pixel_value}')

# Connect the event handler to the figure
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()










