import numpy as np

from conftest import assert_structure
from devito import (
    Buffer, Eq, Function, Grid, Inc, Operator, SubDimension, SubDomain, TimeFunction,
    solve
)
from devito.ir.iet import retrieve_iteration_tree
from devito.ir.support.properties import PARALLEL


def test_issue_1725():

    class ToyPMLLeft(SubDomain):
        name = 'toypmlleft'

        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('left', 2)}

    class ToyPMLRight(SubDomain):
        name = 'toypmlright'

        def define(self, dimensions):
            x, y = dimensions
            return {x: x, y: ('right', 2)}

    subdomains = [ToyPMLLeft(), ToyPMLRight()]
    grid = Grid(shape=(20, 20), subdomains=subdomains)

    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

    eqns = [Eq(u.forward, solve(u.dt2 - u.laplace, u.forward), subdomain=sd)
            for sd in subdomains]

    op = Operator(eqns, opt='fission')

    # Note the `x` loop is fissioned, so now both loop nests can be collapsed
    # for maximum parallelism
    assert_structure(op, ['t,x,y', 't,x,y'], 't,x,y,x,y')


def test_nofission_as_unprofitable():
    """
    Test there's no fission if no increase in number of collapsible loops.
    """
    grid = Grid(shape=(20, 20))
    x, y = grid.dimensions
    t = grid.stepping_dim

    yl = SubDimension.left(name='yl', parent=y, thickness=4)
    yr = SubDimension.right(name='yr', parent=y, thickness=4)

    u = TimeFunction(name='u', grid=grid)

    eqns = [Eq(u.forward, u[t + 1, x, y + 1] + 1.).subs(y, yl),
            Eq(u.forward, u[t + 1, x, y - 1] + 1.).subs(y, yr)]

    op = Operator(eqns, opt='fission')

    assert_structure(op, ['t,x,y', 't,x,y'], 't,x,y,y')


def test_nofission_as_illegal():
    """
    Test there's no fission if dependencies would break.
    """
    grid = Grid(shape=(20, 20))
    x, y = grid.dimensions

    f = Function(name='f', grid=grid, dimensions=(y,), shape=(20,))
    u = TimeFunction(name='u', grid=grid)
    v = TimeFunction(name='v', grid=grid)

    eqns = [Inc(f, v + 1.),
            Eq(u.forward, f[y + 1] + 1.)]

    op = Operator(eqns, opt='fission')

    assert_structure(op, ['t,x,y', 't,x,y'], 't,x,y,y')


def test_fission_partial():
    """
    Test there's no fission if no increase in number of collapsible loops.
    """
    grid = Grid(shape=(20, 20))
    x, y = grid.dimensions
    t = grid.stepping_dim

    yl = SubDimension.left(name='yl', parent=y, thickness=4)
    yr = SubDimension.right(name='yr', parent=y, thickness=4)

    u = TimeFunction(name='u', grid=grid)

    eqns = [Eq(u.forward, u[t + 1, x, y + 1] + 1.).subs(y, yl),
            Eq(u.forward, u[t + 1, x, y - 1] + 1.).subs(y, yr),
            Eq(u.forward, u[t + 1, x, y] + 1.)]

    op = Operator(eqns, opt='fission')

    assert_structure(op, ['t,x,y', 't,x,y', 't,x,y'], 't,x,y,y,x,y')


def test_issue_1921():
    space_order = 4
    grid = Grid(shape=(8, 8), dtype=np.int32)

    f = Function(name='f', grid=grid, space_order=space_order)
    g = TimeFunction(name='g', grid=grid, space_order=space_order)
    g1 = TimeFunction(name='g', grid=grid, space_order=space_order)

    f.data[:] = np.arange(8*8).reshape((8, 8))

    t, x, y = g.dimensions
    ymin = y.symbolic_min

    eqns = []
    eqns.append(Eq(g.forward, f + g))
    for i in range(space_order//2):
        eqns.append(Eq(g[t+t.spacing, x, ymin-i], g[t+t.spacing, x, ymin+i]))

    op0 = Operator(eqns)
    op1 = Operator(eqns, opt='fission')

    assert_structure(op1, ['t,x,y', 't,x'], 't,x,y,x')

    op0.apply(time_m=1, time_M=5)
    op1.apply(time_m=1, time_M=5, g=g1)

    assert np.all(g.data == g1.data)


def test_buffer1_fissioning():
    """
    Tests an edge case whereby inability to spot the equivalence of
    `f.forward`/`backward` and `f` when using `Buffer(1)` would cause
    data dependance analysis to incorrectly determine that `x` and `y`
    loops should be sequential.
    """
    so = 2

    class SD0(SubDomain):
        name = 'sd0'

        def __init__(self, so, **kwargs):
            self.so = so
            super().__init__(**kwargs)

        def define(self, dimensions):
            map_d = {d: d for d in dimensions}
            map_d[dimensions[-1]] = ('left', self.so)
            return map_d

    grid = Grid(shape=(101, 101, 101))
    z = grid.dimensions[-1]

    sd0 = SD0(so, grid=grid)
    iz = sd0.dimensions[-1]

    f0 = TimeFunction(name='f0', grid=grid,
                      space_order=so, save=Buffer(1))
    f1 = TimeFunction(name='f1', grid=grid,
                      space_order=so, save=Buffer(1))

    f2 = TimeFunction(name='f2', grid=grid,
                      space_order=so, save=Buffer(1))
    f3 = TimeFunction(name='f3', grid=grid,
                      space_order=so, save=Buffer(1))

    # Free-surface like conditions plus derivative creates a dependence
    eq0 = Eq(f0.backward.subs(z, iz-so), -f0.backward.subs(z, so-iz))
    eq1 = Eq(f1.backward.subs(z, iz-so), -f1.backward.subs(z, so-iz))

    eq2 = Eq(f2, f0.dx + f2)
    eq3 = Eq(f3, f1.dy + f3)

    eqs = [eq0, eq1, eq2, eq3]

    op = Operator(eqs)

    trees = retrieve_iteration_tree(op)

    # If the equivalence of f0.backward and f0 are not spotted by the
    # compiler, data dependence analysis will determine x and y loops
    # to be sequential.
    for tree in trees:
        for i in range(1, 3):
            assert PARALLEL in tree[i].properties

    # If parallelisation failed, then there will also be no blocking on
    # x and y
    assert 'x0_blk0_size' in str(op.parameters)
    assert 'y0_blk0_size' in str(op.parameters)
