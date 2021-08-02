from conftest import assert_structure
from devito import (Eq, Inc, Grid, Function, TimeFunction, SubDimension, SubDomain,
                    Operator, solve)


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
    assert_structure(op, ['t,x,i1y', 't,x,i2y'], 't,x,i1y,x,i2y')


def test_nofission_as_unprofitable():
    """
    Test there's no fission if not gonna increase number of collapsable loops.
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

    assert_structure(op, ['t,x,yl', 't,x,yr'], 't,x,yl,yr')


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
    Test there's no fission if not gonna increase number of collapsable loops.
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

    assert_structure(op, ['t,x,yl', 't,x,yr', 't,x,y'], 't,x,yl,yr,x,y')
