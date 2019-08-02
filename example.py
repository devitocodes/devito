from devito import TimeFunction, Grid, Operator, Eq

grid = Grid((4, 4))
v = TimeFunction(name='v', grid=grid, space_order=2)

op = Operator(Eq(v, v.dxl + v.dxr - v.dyr - v.dyl))
