import numpy as np

from conftest import assert_structure
from devito import (
    Eq, Function, Grid, Inc, Operator, SubDimension, SubDomain, TimeFunction, solve
)
from devito.types import Symbol


class TestFissionForParallelism:

    def test_issue_1725(self):

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

    def test_nofission_as_unprofitable(self):
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

    def test_nofission_as_illegal(self):
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

    def test_fission_partial(self):
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

    def test_issue_1921(self):
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


class TestFissionForPressure:

    def test_basic(self):
        grid = Grid(shape=(20, 20))

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(u.forward, u + 1),
                Eq(v.forward, v + 1)]

        op = Operator(eqns, opt=('fuse', 'fission', {'openmp': False,
                                                     'fiss-press-size': 1}))

        assert_structure(op, ['t,x,y', 't,x,y'], 't,x,y,y')

    def test_nofission_as_illegal(self):
        grid = Grid(shape=(20, 20))

        s = Symbol(name='s', dtype=grid.dtype)
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(s, u + v),
                Eq(u.forward, u + 1),
                Eq(v.forward, v + s + 1)]

        op = Operator(eqns, opt=('fuse', 'fission', {'openmp': False,
                                                     'fiss-press-size': 1,
                                                     'fiss-press-ratio': 1}))

        assert_structure(op, ['t,x,y'], 't,x,y')

    def test_ge_threshold_ratio(self):
        grid = Grid(shape=(20, 20))

        f0 = Function(name='f0', grid=grid)
        f1 = Function(name='f1', grid=grid)
        w0 = Function(name='w0', grid=grid)
        w1 = Function(name='w1', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(u.forward, u + f0 + w0 + w1 + 1.),
                Eq(v.forward, v + f1 + w0 + w1 + 1.)]

        op = Operator(eqns, opt=('fuse', 'fission', {'openmp': False,
                                                     'fiss-press-size': 1}))

        # There are four Functions in both the first and the second Eq
        # There are two Functions, w0 and w1, shared by both Eqs
        # Hence, given that the default fiss-press-ratio is 2...
        assert op.FISS_PRESS_RATIO == 2
        # ... we are >= threshold, hence we expect fissioning

        assert_structure(op, ['t,x,y', 't,x,y'], 't,x,y,y')

    def test_lt_threshold_ratio(self):
        grid = Grid(shape=(20, 20))

        w0 = Function(name='w0', grid=grid)
        w1 = Function(name='w1', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(u.forward, u + w0 + w1 + 1.),
                Eq(v.forward, v + w0 + w1 + 1.)]

        op = Operator(eqns, opt=('fuse', 'fission', {'openmp': False,
                                                     'fiss-press-size': 1}))

        # There are three Functions in both the first and the second Eq
        # There are two Functions, w0 and w1, shared by both Eqs
        # Hence, given that the default fiss-press-ratio is 2...
        assert op.FISS_PRESS_RATIO == 2
        # ... we are < threshold, hence we don't expect fissioning

        assert_structure(op, ['t,x,y'], 't,x,y')
