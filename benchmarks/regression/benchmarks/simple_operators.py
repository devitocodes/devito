from devito import Grid, TimeFunction, Eq, Operator


repeat = 3


def time_basic():
    grid = Grid(shape=(16, 16, 16))

    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1))

    op.apply(time_M=10)
