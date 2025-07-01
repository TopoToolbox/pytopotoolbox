import topotoolbox as topo
import numpy as np
import scipy.sparse as sp
import scipy.optimize as op

# define all objects 
dem = topo.load_dem('perfectworld')
fd = topo.FlowObject(dem)
s = topo.StreamObject(fd, threshold = 250000)
tau = 0.75 # quantile of the data you are targeting

# get node attribute list with elevation values
z = s.ezgetnal(dem, dtype = 'double') # elevation values of the dem
nr = z.size

# Quantcarve algorithm (equation A12)
d = s.upstream_distance() # had to use downstream, so indices and source/target are flipped in the subsequent code
f = np.vstack((tau*np.ones((nr, 1)), (1-tau)*np.ones((nr,1)), np.zeros((nr,1))))

# Constraints 
# gradient
dd = np.array(1/(d[s.target]-d[s.source])) # cellsize
gradient = (sp.coo_matrix((dd, (s.source, s.source)), shape=(nr, nr)) - sp.coo_matrix((dd, (s.source, s.target)), shape=(nr, nr))).tocsr() 
a_matrix = sp.hstack([sp.csr_matrix((nr, nr)), sp.csr_matrix((nr, nr)), -gradient])
b = np.zeros((nr,1)) # min gradient
b[s.source] = 0.01
b = sp.csr_matrix((nr,1)).toarray() 

# bounds constraints
lb = np.vstack([np.zeros((nr, 1)), np.zeros((nr, 1)), -float('inf')*np.ones((nr,1))]) # 2D array 
lb = (lb.flatten()) # convert to 1D list
ub = np.inf*np.ones(3*nr)

# equality constraint 
identity = sp.eye(nr)
fixedoutlet = False
if fixedoutlet:
    outlet = s.streampoi('outlets')
    p = sp.diags((~outlet).astype(int), 0, shape = (nr,nr))
    a_eq = sp.hstack([p, -p, identity])
else: 
    a_eq = sp.hstack([identity, -identity, identity])

b_eq = z

# solve linear programming problem
bhat = op.linprog(f, a_matrix, b, a_eq, b_eq, bounds = list(zip(lb, ub)))
zs = bhat['x'][2*nr:]

