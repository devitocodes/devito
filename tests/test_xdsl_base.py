from devito import Grid, Function, Eq, XDSLOperator, Operator
from devito.ir.ietxdsl.xdsl_passes import transform_devito_xdsl_string
# flake8: noqa
from devito.operator.xdsl_operator import XDSLOperator


def test_create_xdsl_operator():

    # Define a simple Devito Operator
    grid = Grid(shape=(3,))
    u = Function(name='u', grid=grid)
    eq = Eq(u, u + 1)
    xdsl_op = XDSLOperator([eq])
    xdsl_op.__class__ = XDSLOperator
    xdsl_op.apply()

    op = Operator([eq])
    op.apply()

    # import pdb;pdb.set_trace()
    # TOFIX to add proper test
    # assert(str(op.ccode) == xdsl_op.ccode)

    from xdsl.printer import Printer
    printer = Printer(target=Printer.Target.MLIR)
    printer.print_op(xdsl_op)
    assert False
    # import pdb;pdb.set_trace()
    # printer.print_op(mod)
