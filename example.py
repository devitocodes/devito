from devito.ir import ietxdsl
from devito import Grid, Function, TimeFunction, Eq, Operator, Constant

if __name__ == '__main__':

    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    #op.apply(time_M=5)

    module = ietxdsl.transform_devito_to_iet_ssa(op)

    from xdsl.printer import Printer
    Printer(target=Printer.Target.MLIR).print(module)