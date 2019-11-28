from mpi4py import MPI
import numpy as np
from math import floor

from devito import (Grid, Function, TimeFunction, Eq, solve, Operator, SubDomainSet,
                    Dimension, configuration)

configuration['mpi'] = True

n_domains = 5

class Inner(SubDomainSet):
    name = 'inner'

bounds_xm = np.zeros((n_domains,), dtype=np.int32)
bounds_xM = np.zeros((n_domains,), dtype=np.int32)
bounds_ym = np.zeros((n_domains,), dtype=np.int32)
bounds_yM = np.zeros((n_domains,), dtype=np.int32)

for j in range(0, n_domains):
    bounds_xm[j] = j
    bounds_xM[j] = j
    bounds_ym[j] = j
    bounds_yM[j] = 2*n_domains-1-j

bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

inner_sd = Inner(N=n_domains, bounds=bounds)

grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

f = TimeFunction(name='f', grid=grid, dtype=np.int32)
f.data[:] = 0

stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
             subdomain=grid.subdomains['inner'])

op = Operator(stencil)
op(time_m=0, time_M=9, dt=1)
result = f.data[0]

expected = np.zeros((10, 10), dtype=np.int32)
for j in range(0, n_domains):
    expected[j, bounds_ym[j]:n_domains-bounds_yM[j]] = 10

#assert((np.array(result) == expected).all())

from IPython import embed; embed()
