from devito import Grid, TimeFunction, Eq, XDSLOperator


def test_xdsl_I():
    # Define a simple Devito Operator
    grid = Grid(shape=(3,))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_II():
    # Define a simple Devito Operator
    grid = Grid(shape=(4, 4))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()


def test_xdsl_III():
    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5, 5))
    u = TimeFunction(name='u', grid=grid)

    eq = Eq(u.forward, u + 1)
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
    assert (u.data[1, :] == 1.).all()
    assert (u.data[0, :] == 2.).all()