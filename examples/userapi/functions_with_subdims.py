import numpy as np

from devito import Grid, Function, Eq, Operator
from devito.data import Decomposition

grid = Grid(shape=(11, 11))

subdims = grid.subdomains['interior'].dimensions

xdecomp, ydecomp = grid._distributor.decomposition

sd_decomp = (xdecomp.reshape(slice(1,-1)), ydecomp.reshape(slice(1,-1)))

f = Function(name='f', dimensions=subdims, shape=grid.subdomains['interior'].shape, space_order=0)

eq = Eq(f, f+1)

op = Operator(eq)

from IPython import embed; embed()
