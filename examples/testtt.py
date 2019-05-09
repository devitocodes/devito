from devito import Grid, TimeFunction, Eq, Operator

grid = Grid(shape=(4, 4))
u = TimeFunction(name='u', grid=grid, save=3)
op = Operator(Eq(u.forward, u + 1),dse='advanced')
print(op)
