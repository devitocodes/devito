from devito import Grid, TimeFunction, Eq, Operator


repeat = 3


def time_basic():
    grid = Grid(shape=(400, 400, 400))

    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1))

    op.apply(time_M=100)


def time_laplacian():
    grid = Grid(shape=(400, 400, 400))

    f = TimeFunction(name='f', grid=grid, space_order=2)

    op = Operator(Eq(f.forward, 1e-8*(f.laplace + 1)))

    op.apply(time_M=100)
