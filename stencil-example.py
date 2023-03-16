from devito.ir import ietxdsl
from devito import Grid, Function, TimeFunction, Eq, Operator, Constant, XDSLOperator, norm, solve
import numpy as np

from devito import configuration

from xdsl.printer import Printer

from devito.ir.ietxdsl import cluster_to_ssa

configuration['log-level'] = 'DEBUG'

from xdsl.pattern_rewriter import PatternRewriteWalker, GreedyRewritePatternApplier

if __name__ == '__main__':

    nx, ny = 300, 300
    nt = 2
    nu = .5
    dx = 2. / (nx - 1)
    dy = 2. / (ny - 1)
    sigma = .25
    dt = sigma * dx * dy / nu

    so = 2
    to = 2

    # Initialise u
    init_value = 10

    # Field initialization
    grid = Grid(shape=(nx, ny))
    u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
    u.data[:, :, :] = 0
    u.data[:, int(nx/2), int(nx/2)] = init_value
    u.data[:, int(nx/2), -int(nx/2)] = -init_value

    # Create an equation with second-order derivatives
    a = Constant(name='a')
    eq = Eq(u.dt2, a*u.laplace + 0.01)
    stencil = solve(eq, u.forward)
    eq0 = Eq(u.forward, stencil)
    #xop = XDSLOperator([eq])
    #xop.apply(time_M=steps)
    #xdsl_data: np.array = u.data_with_halo.copy()
    #xdsl_norm = norm(u)

    u.data[:,:] = 5
    op = Operator([eq0])

    cluster_to_ssa.ExtractDevitoStencilConversion(op)

    #op.apply(time_M=steps)
    #orig_data: np.array = u.data_with_halo.copy() 
    #orig_norm = norm(u)


    print("orig={}, xdsl={}".format(xdsl_norm, orig_norm))
    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


    #module = ietxdsl.transform_devito_to_iet_ssa(op)

    #p = Printer(target=Printer.Target.MLIR)
    #p.print(module)

    #print("\n\nAFTER REWRITE:\n")

    #ietxdsl.iet_to_standard_mlir(module)

    #p = Printer(target=Printer.Target.MLIR)
    #p.print(module)