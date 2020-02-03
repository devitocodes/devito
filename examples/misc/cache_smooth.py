import numpy as np
from mpi4py import MPI
from scipy.ndimage import gaussian_filter

from devito import Grid, Function
from devito.builtins import assign, gaussian_smooth
from devito import configuration
from devito.tools import as_tuple

from devito import Constant

configuration['mpi'] = True


grid = Grid(shape=(100, 100))

f = Function(name='f', grid=grid, dtype=np.int32)
g = Function(name='g', grid=grid, dtype=np.int32)

assign(f, 3*f*g+5*g, save=True)
assign(g, 2*g*f+1*f, save=True)

a = Constant(name='a')
a.data = 2.0
b = Constant(name='b')
b.data = 3.0

print(type(a) == type(b))
#print('finished')
from IPython import embed; embed()

#smoother(f, sigma=5)
#smoother(f, sigma=5)
#smoother(f, sigma=5)
