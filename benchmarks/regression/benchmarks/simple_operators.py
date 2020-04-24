from devito import Grid, TimeFunction, Eq, Operator, norm


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


def time_norm():
    grid = Grid(shape=(400, 400, 400))

    f = TimeFunction(name='f', grid=grid, space_order=2)

    norm(f)
