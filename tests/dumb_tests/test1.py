from devito import Grid, TimeFunction,  Eq, Operator

grid = Grid(shape=(3, 3))
u = TimeFunction(name='u', grid=grid)
# u.data
eq = Eq(u.forward, u+1)
op = Operator(eq)

print(op)