from devito.ir import ietxdsl
from devito import Grid, Function, TimeFunction, Eq, Operator, Constant, XDSLOperator, norm
import numpy as np

from devito import configuration

from xdsl.printer import Printer

configuration['log-level'] = 'DEBUG'

from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier

if __name__ == '__main__':

    grid = Grid(shape=(300, 300))
    steps = 5_000

    u = TimeFunction(name='u', grid=grid)
    u.data[:,:] = 5
    eq = Eq(u.forward, u + 0.1)
    xop = XDSLOperator([eq])
    xop.apply(time_M=steps)
    xdsl_data: np.array = u.data_with_halo.copy()
    xdsl_norm = norm(u)

    u.data[:,:] = 5
    op = Operator([eq])
    op.apply(time_M=steps)
    orig_data: np.array = u.data_with_halo.copy() 
    orig_norm = norm(u)


    print("orig={}, xdsl={}".format(xdsl_norm, orig_norm))
    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


    #module = ietxdsl.transform_devito_to_iet_ssa(op)

    #p = Printer(target=Printer.Target.MLIR)
    #p.print(module)

    #print("\n\nAFTER REWRITE:\n")

    #ietxdsl.iet_to_standard_mlir(module)

    #p = Printer(target=Printer.Target.MLIR)
    #p.print(module)