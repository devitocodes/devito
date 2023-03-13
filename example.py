from devito.ir import ietxdsl
from devito import Grid, Function, TimeFunction, Eq, Operator, Constant

from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier

if __name__ == '__main__':

    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    #op.apply(time_M=5)

    module = ietxdsl.transform_devito_to_iet_ssa(op)

    from xdsl.printer import Printer
    p = Printer(target=Printer.Target.MLIR)
    p.print(module)

    print("\n\nAFTER REWRITE:\n")

    ietxdsl.iet_to_standard_mlir(module)

    p = Printer(target=Printer.Target.MLIR)
    p.print(module)