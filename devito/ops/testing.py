from devito import Eq, Grid, TimeFunction, Operator

grid = Grid(shape=(3, 3))
u = TimeFunction(name='u', grid=grid)
u.data

eq = Eq(u.forward, u+1)
op = Operator(eq)

op.apply(time=1)

a = 1

from devito import *
print_defaults()
print_state()

a = 2
