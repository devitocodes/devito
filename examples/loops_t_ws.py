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

bounds_xm = Function(name='bounds_xm', shape=(n_domains, ), dimensions=(n, ), dtype=np.int32)
bounds_xM = Function(name='bounds_xM', shape=(n_domains, ), dimensions=(n, ), dtype=np.int32)
bounds_ym = Function(name='bounds_ym', shape=(n_domains, ), dimensions=(n, ), dtype=np.int32)
bounds_yM = Function(name='bounds_yM', shape=(n_domains, ), dimensions=(n, ), dtype=np.int32)

for j in range(0,n_domains):
    bounds_xm.data[j] = j
    bounds_xM.data[j] = n_domains-1-j
    bounds_ym.data[j] = floor(j/2)
    bounds_yM.data[j] = floor(j/2)
    
inner_sd = Inner(n_domains=n_domains, extent=(bounds_xm.data, bounds_xM.data,
                                              bounds_ym.data, bounds_yM.data))
    
grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))
t = grid.time_dim
x, y = grid.dimensions

f = TimeFunction(name='f', grid=grid)
f.data[:] = 0.0
eq = Eq(f.dt, 1)

stencil = Eq(f.forward, solve(eq, f.forward),
             implicit_dims=[t, n, inner_sd.dimensions[0], inner_sd.dimensions[1]], 
             subdomain=grid.subdomains['inner'])

eq_xm = Eq(inner_sd.dimensions[0].thickness[0][0], bounds_xm[n], implicit_dims=[t, n])
eq_xM = Eq(inner_sd.dimensions[0].thickness[1][0], bounds_xM[n], implicit_dims=[t, n])
eq_ym = Eq(inner_sd.dimensions[1].thickness[0][0], bounds_ym[n], implicit_dims=[t, n])
eq_yM = Eq(inner_sd.dimensions[1].thickness[1][0], bounds_yM[n], implicit_dims=[t, n])

op = Operator([eq_xm, eq_xM, eq_ym, eq_yM, stencil])

#print(op.ccode)
op(time_m=0, time_M=0, dt=1)

from IPython import embed; embed()
