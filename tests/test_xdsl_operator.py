from devito import Grid, TimeFunction, Eq, XDSLOperator, Operator
from devito.operator.xdsl_operator import XDSLOperator
# flake8: noqa


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    xdsl_op = XDSLOperator([eq])
    xdsl_op.apply(time_M=5)

    op = XDSLOperator([eq])
    op.apply(time_M=5)


def test_opt_xdsl():
    # Following Devito's path for the moment
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq], opt='xdsl')
    op.apply(time_M=5)
    import pdb;pdb.set_trace()