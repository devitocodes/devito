import numpy as np
from devito import Grid, SubDomain, Dimension
from devito import Function, TimeFunction, Eq, Constant, Operator, solve
from devito.types import SubDomains

from math import floor

n_domains = 10
n = Dimension(name = 'n')

class Inner(SubDomains):
    name = 'inner'
    def define(self, dimensions):
        return {d: ('middle', 0, 0) for d in dimensions}

bounds_xm = np.zeros((n_domains,), dtype=np.int32)
bounds_xM = np.zeros((n_domains,), dtype=np.int32)
bounds_ym = np.zeros((n_domains,), dtype=np.int32)
bounds_yM = np.zeros((n_domains,), dtype=np.int32)
    
for j in range(0,n_domains):
    bounds_xm[j] = j
    bounds_xM[j] = n_domains-1-j
    bounds_ym[j] = floor(j/2)
    bounds_yM[j] = floor(j/2)
    
extent=(bounds_xm, bounds_xM, bounds_ym, bounds_yM)

inner_sd = Inner(n_domains=n_domains, extent=extent)

grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))
t = grid.time_dim
x, y = grid.dimensions

f = TimeFunction(name='f', grid=grid)
f.data[:] = 0.0

stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
             subdomain=grid.subdomains['inner'])

op = Operator(stencil)

#print(op.ccode)
#op(time_m=0, time_M=0, dt=1)

from IPython import embed; embed()
