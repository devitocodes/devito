from devito import Grid, TimeFunction, Eq, Operator


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)

    op = Operator([eq], opt='xdsl')
    op.apply(time_M=5)


def test_opt_xdsl():
    # Following Devito's path for the moment
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq], opt='xdsl')
    # op = Operator([eq], opt='advanced')
    op.apply(time_M=5)
