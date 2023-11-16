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
