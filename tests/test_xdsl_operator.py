from devito import Grid, TimeFunction, Eq, XDSLOperator
from devito.ir.ietxdsl.xdsl_passes import transform_devito_xdsl_string
# flake8: noqa
from devito.operator.xdsl_operator import XDSLOperator


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    xdsl_op = XDSLOperator([eq])
    xdsl_op.__class__ = XDSLOperator
    xdsl_op.apply(time_M=5)

    print(str(xdsl_op.ccode))

