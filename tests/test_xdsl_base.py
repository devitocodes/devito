from devito import Grid, TimeFunction, Eq, XDSLOperator


def test_simple_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u + 1)
    
    op = XDSLOperator([eq], opt=None)
    op.apply(time_M=1)
