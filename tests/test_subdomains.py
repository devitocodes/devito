import pytest
import numpy as np
from math import floor

from sympy import sin, tan

from conftest import opts_tiling, assert_structure
from devito import (ConditionalDimension, Constant, Grid, Function, TimeFunction,
                    Eq, solve, Operator, SubDomain, SubDomainSet)
from devito.ir import FindNodes, Expression, Iteration
from devito.tools import timed_region


class TestSubdomains(object):
    """
    Class for testing SubDomains
    """

    def test_subdomain_dim(self):
        """
        Test that all dimensions including ones used as an expression
        are replaced by the subdimension dimensions.
        """
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()
        grid = Grid(shape=(10, 10), subdomains=(s_d0,))
        x, y = grid.dimensions
        x1, y1 = s_d0.dimensions
        f = Function(name='f', grid=grid, dtype=np.int32)

        eq0 = Eq(f, x*f+y, subdomain=grid.subdomains['d0'])
        with timed_region('x'):
            expr = Operator._lower_exprs([eq0], options={})[0]
        assert expr.rhs == x1 * f[x1 + 1, y1 + 1] + y1

    def test_multiple_middle(self):
        """
        Test Operator with two basic 'middle' subdomains defined.
        """
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()

        class sd1(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 6, 1), y: ('middle', 1, 1)}
        s_d1 = sd1()

        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1))

        f = Function(name='f', grid=grid, dtype=np.int32)

        eq0 = Eq(f, f+1, subdomain=grid.subdomains['d0'])
        eq1 = Eq(f, f+2, subdomain=grid.subdomains['d1'])

        Operator([eq0, eq1])()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        assert((np.array(f.data) == expected).all())

    def test_shape(self):
        class sd0(SubDomain):
            name = 'd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 6), y: ('middle', 1, 1)}
        s_d0 = sd0()

        class sd1(SubDomain):
            name = 'd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('right', 4), y: ('left', 2)}
        s_d1 = sd1()

        class sd2(SubDomain):
            name = 'd2'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('left', 3), y: ('middle', 1, 2)}
        s_d2 = sd2()

        grid = Grid(shape=(10, 10), subdomains=(s_d0, s_d1, s_d2))

        assert grid.subdomains['domain'].shape == (10, 10)
        assert grid.subdomains['interior'].shape == (8, 8)

        assert grid.subdomains['d0'].shape == (3, 8)
        assert grid.subdomains['d1'].shape == (4, 2)
        assert grid.subdomains['d2'].shape == (3, 7)

    def test_definitions(self):

        class sd0(SubDomain):
            name = 'sd0'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('right', 10)}

        class sd1(SubDomain):
            name = 'sd1'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('left', 10)}

        class sd2(SubDomain):
            name = 'sd2'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: y}

        class sd3(SubDomain):
            name = 'sd3'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 0, 0)}

        sd_def0 = sd0()
        sd_def1 = sd1()
        sd_def2 = sd2()
        sd_def3 = sd3()

        grid = Grid(shape=(10, 10), extent=(10, 10),
                    subdomains=(sd_def0, sd_def1, sd_def2, sd_def3))
        u0 = Function(name='u0', grid=grid)
        u1 = Function(name='u1', grid=grid)
        u2 = Function(name='u2', grid=grid)
        u3 = Function(name='u3', grid=grid)
        eq0 = Eq(u0, u0+1, subdomain=grid.subdomains['sd0'])
        eq1 = Eq(u1, u1+1, subdomain=grid.subdomains['sd1'])
        eq2 = Eq(u2, u2+1, subdomain=grid.subdomains['sd2'])
        eq3 = Eq(u3, u3+1, subdomain=grid.subdomains['sd3'])
        Operator([eq0, eq1, eq2, eq3])()

        assert u0.data.all() == u1.data.all() == u2.data.all() == u3.data.all()


class TestMultiSubDomain(object):

    @pytest.mark.parametrize('opt', opts_tiling)
    def test_iterate_NDomains(self, opt):
        """
        Test that a set of subdomains are iterated upon correctly.
        """

        n_domains = 10

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = n_domains-1-j
            bounds_ym[j] = floor(j/2)
            bounds_yM[j] = floor(j/2)

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil, opt=opt)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, bounds_ym[j]:n_domains-bounds_yM[j]] = 10

        assert((np.array(result) == expected).all())

    def test_multi_eq(self):
        """
        Test SubDomainSet functionality when multiple equations are
        present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        class MySubdomains(SubDomainSet):
            name = 'mydomains'

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = 1
        bounds_yM = 1
        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)
        my_sd = MySubdomains(N=n_domains, bounds=bounds)
        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd, ))

        assert(grid.subdomains['mydomains'].shape == ((3, 8), (3, 8)))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)
        h = Function(name='h', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains'])
        eq2 = Eq(g, g+1)
        eq3 = Eq(h, h+2, subdomain=grid.subdomains['mydomains'])

        op = Operator([eq1, eq2, eq3])
        op.apply()

        expected1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)
        expected2 = np.full((10, 10), 1, dtype=np.int32)
        expected3 = 2*expected1

        assert((np.array(f.data) == expected1).all())
        assert((np.array(g.data) == expected2).all())
        assert((np.array(h.data) == expected3).all())

        # Also make sure the Functions carrying the subdomain bounds are
        # unique -- see issue #1474
        exprs = FindNodes(Expression).visit(op)
        reads = set().union(*[e.reads for e in exprs])
        assert len(reads) == 7  # f, g, h, xi_n_m, xi_n_M, yi_n_m, yi_n_M

    def test_multi_sets(self):
        """
        Check functionality for when multiple subdomain sets are present.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        class MySubdomains1(SubDomainSet):
            name = 'mydomains1'

        class MySubdomains2(SubDomainSet):
            name = 'mydomains2'

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(1)
        bounds_yM = int(Ny/2+1)
        bounds1 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(Ny/2+1)
        bounds_yM = int(1)
        bounds2 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        my_sd1 = MySubdomains1(N=n_domains, bounds=bounds1)
        my_sd2 = MySubdomains2(N=n_domains, bounds=bounds2)

        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd1, my_sd2))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+1, subdomain=grid.subdomains['mydomains1'])
        eq2 = Eq(g, g+2, subdomain=grid.subdomains['mydomains2'])

        op = Operator([eq1, eq2])
        op.apply()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 1, 1, 1, 0, 0, 2, 2, 2, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

        assert((np.array(f.data[:]+g.data[:]) == expected).all())

    @pytest.mark.parallel(mode=[(4, 'basic'), (4, 'overlap')])
    def test_subdomainset_mpi(self):

        n_domains = 5

        class Inner(SubDomainSet):
            name = 'inner'

        bounds_xm = np.zeros((n_domains,), dtype=np.int32)
        bounds_xM = np.zeros((n_domains,), dtype=np.int32)
        bounds_ym = np.zeros((n_domains,), dtype=np.int32)
        bounds_yM = np.zeros((n_domains,), dtype=np.int32)

        for j in range(0, n_domains):
            bounds_xm[j] = j
            bounds_xM[j] = j
            bounds_ym[j] = j
            bounds_yM[j] = 2*n_domains-1-j

        bounds = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        inner_sd = Inner(N=n_domains, bounds=bounds)

        grid = Grid(extent=(10, 10), shape=(10, 10), subdomains=(inner_sd, ))

        assert(grid.subdomains['inner'].shape == ((10, 1), (8, 1), (6, 1),
                                                  (4, 1), (2, 1)))

        f = TimeFunction(name='f', grid=grid, dtype=np.int32)
        f.data[:] = 0

        stencil = Eq(f.forward, solve(Eq(f.dt, 1), f.forward),
                     subdomain=grid.subdomains['inner'])

        op = Operator(stencil)
        op(time_m=0, time_M=9, dt=1)
        result = f.data[0]

        fex = Function(name='fex', grid=grid)
        expected = np.zeros((10, 10), dtype=np.int32)
        for j in range(0, n_domains):
            expected[j, j:10-j] = 10
        fex.data[:] = np.transpose(expected)

        assert((np.array(result) == np.array(fex.data[:])).all())

    def test_multi_sets_eq(self):
        """
        Check functionality for when multiple subdomain sets are present, each
        with multiple equations.
        """

        Nx = 10
        Ny = Nx
        n_domains = 2

        class MySubdomains1(SubDomainSet):
            name = 'mydomains1'

        class MySubdomains2(SubDomainSet):
            name = 'mydomains2'

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(1)
        bounds_yM = int(Ny/2+1)
        bounds1 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        bounds_xm = np.array([1, Nx/2+1], dtype=np.int32)
        bounds_xM = np.array([Nx/2+1, 1], dtype=np.int32)
        bounds_ym = int(Ny/2+1)
        bounds_yM = int(1)
        bounds2 = (bounds_xm, bounds_xM, bounds_ym, bounds_yM)

        my_sd1 = MySubdomains1(N=n_domains, bounds=bounds1)
        my_sd2 = MySubdomains2(N=n_domains, bounds=bounds2)

        grid = Grid(extent=(Nx, Ny), shape=(Nx, Ny), subdomains=(my_sd1, my_sd2))

        f = Function(name='f', grid=grid, dtype=np.int32)
        g = Function(name='g', grid=grid, dtype=np.int32)

        eq1 = Eq(f, f+2, subdomain=grid.subdomains['mydomains1'])
        eq2 = Eq(g, g+2, subdomain=grid.subdomains['mydomains2'])
        eq3 = Eq(f, f-1, subdomain=grid.subdomains['mydomains1'])
        eq4 = Eq(g, g+1, subdomain=grid.subdomains['mydomains2'])

        op = Operator([eq1, eq2, eq3, eq4])
        op.apply()

        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                             [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                             [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                             [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                             [0, 1, 1, 1, 0, 0, 3, 3, 3, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)

        assert((np.array(f.data[:]+g.data[:]) == expected).all())

    def test_issue_1761(self):
        """
        MFE for issue #1761.
        """

        class DummySubdomains(SubDomainSet):
            name = 'dummydomain'
        dummy = DummySubdomains(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(10, 10), subdomains=(dummy,))

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)
        theta = Function(name='theta', grid=grid)
        phi = Function(name='phi', grid=grid)

        eqns = [Eq(f.forward, f*sin(phi), subdomain=grid.subdomains['dummydomain']),
                Eq(g.forward, g*sin(theta), subdomain=grid.subdomains['dummydomain'])]

        op = Operator(eqns)

        # Make sure it jit-compiles
        op.cfunction

        assert_structure(op, ['x,y', 't,n0', 't,n0,xi2,yi2'], 'x,y,t,n0,xi2,yi2')

    def test_issue_1761_b(self):
        """
        Follow-up issue emerged after patching #1761. The thicknesses assigments
        were missing before the third equation.

        Further improvements have enabled fusing the third equation with the first
        one, since this is perfectly legal (just like what happens without
        MultiSubDomains in the way).
        """

        class DummySubdomains(SubDomainSet):
            name = 'dummydomain'

        dummy = DummySubdomains(N=1, bounds=(1, 1, 1, 1))

        class DummySubdomains2(SubDomainSet):
            name = 'dummydomain2'

        dummy2 = DummySubdomains2(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(10, 10), subdomains=(dummy, dummy2))

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)
        theta = Function(name='theta', grid=grid)
        phi = Function(name='phi', grid=grid)

        eqns = [Eq(f.forward, f*sin(phi), subdomain=grid.subdomains['dummydomain']),
                Eq(g.forward, g*sin(theta), subdomain=grid.subdomains['dummydomain2']),
                Eq(f.forward, f*tan(phi), subdomain=grid.subdomains['dummydomain'])]

        op = Operator(eqns)

        # Make sure it jit-compiles
        op.cfunction

        assert_structure(op,
                         ['x,y', 't,n0', 't,n0,xi2,yi2', 't,n1', 't,n1,xi3,yi3'],
                         'x,y,t,n0,xi2,yi2,n1,xi3,yi3')

    def test_issue_1761_c(self):
        """
        Follow-up of test test_issue_1761_b. Now there's a data dependence
        between eq0 and eq1, hence they can't be fused.
        """

        class DummySubdomains(SubDomainSet):
            name = 'dummydomain'

        dummy = DummySubdomains(N=1, bounds=(1, 1, 1, 1))

        class DummySubdomains2(SubDomainSet):
            name = 'dummydomain2'

        dummy2 = DummySubdomains2(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(10, 10), subdomains=(dummy, dummy2))

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)
        theta = Function(name='theta', grid=grid)
        phi = Function(name='phi', grid=grid)

        eqns = [Eq(f.forward, f*sin(phi), subdomain=grid.subdomains['dummydomain']),
                Eq(g.forward, g*sin(theta) + f.forward.dx,
                   subdomain=grid.subdomains['dummydomain2']),
                Eq(f.forward, f*tan(phi), subdomain=grid.subdomains['dummydomain'])]

        op = Operator(eqns)

        # Make sure it jit-compiles
        op.cfunction

        assert_structure(op, ['x,y', 't,n0', 't,n0,xi2,yi2',
                              't,n1', 't,n1,xi3,yi3', 't,n0', 't,n0,xi2,yi2'],
                         'x,y,t,n0,xi2,yi2,n1,xi3,yi3,n0,xi2,yi2')

    def test_issue_1761_d(self):
        """
        Follow-up of test test_issue_1761_b. CIRE creates an equation, and the
        creation of the implicit equations needs to be such that no redundant
        thickness assignments are generated.
        """

        class Dummy(SubDomainSet):
            name = 'dummy'

        dummy = Dummy(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(10, 10), subdomains=(dummy,))

        f = TimeFunction(name='f', grid=grid, space_order=4)

        eqn = Eq(f.forward, f.dx.dx + 1, subdomain=grid.subdomains['dummy'])

        op = Operator(eqn)

        # Make sure it jit-compiles
        op.cfunction

        assert_structure(op, ['t,n0', 't,n0,xi2,yi2', 't,n0,xi2,yi2'],
                         't,n0,xi2,yi2,xi2,yi2')

    def test_guarding(self):

        class Dummy(SubDomainSet):
            name = 'dummy'

        dummy = Dummy(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(10, 10), subdomains=(dummy,))
        time = grid.time_dim

        c = Constant(name='c')
        cond_a = ConditionalDimension(name='cond_a', parent=time, condition=c < 1.)
        cond_b = ConditionalDimension(name='cond_b', parent=time, condition=c >= 1.)

        f = TimeFunction(name='f', grid=grid)
        g = TimeFunction(name='g', grid=grid)

        eqns = [Eq(f.forward, f + 1., subdomain=dummy, implicit_dims=[cond_a]),
                Eq(g.forward, g + 1., subdomain=dummy, implicit_dims=[cond_b])]

        op = Operator(eqns)

        # Make sure it jit-compiles
        op.cfunction

        assert_structure(op, ['t', 't,n0', 't,n0,xi2,yi2', 't,n0', 't,n0,xi2,yi2'],
                         't,n0,xi2,yi2,n0,xi2,yi2')

    def test_3D(self):

        class Dummy(SubDomainSet):
            name = 'dummy'

        dummy = Dummy(N=0, bounds=[(), (), (), (), (), ()])

        grid = Grid(shape=(10, 10, 10), subdomains=(dummy,))

        f = TimeFunction(name='f', grid=grid)

        eqn = Eq(f.forward, f.dx + 1, subdomain=grid.subdomains['dummy'])

        op = Operator(eqn)

        # Make sure it jit-compiles
        op.cfunction

        assert_structure(op, ['t,n0', 't,n0,xi20_blk0,yi20_blk0,xi2,yi2,zi2'],
                         't,n0,xi20_blk0,yi20_blk0,xi2,yi2,zi2')

    def test_sequential_implicit(self):
        """
        Make sure the implicit dimensions of the MultiSubDomain define a sequential
        iteration space. This is for performance and potentially for correctness too
        (e.g., canonical openmp loops forbid subiterators, which could potentially be
        required by a MultiSubDomain).
        """

        class Dummy(SubDomainSet):
            name = 'dummy'

        dummy = Dummy(N=0, bounds=[(), (), (), (), (), ()])

        grid = Grid(shape=(10, 10, 10), subdomains=(dummy,))

        f = TimeFunction(name='f', grid=grid, save=10)

        eqn = Eq(f, 1., subdomain=grid.subdomains['dummy'])

        op = Operator(eqn)

        iterations = FindNodes(Iteration).visit(op)
        time, n, x, y, z = iterations
        assert time.is_Sequential
        assert n.is_Sequential
        assert x.is_Parallel
        assert y.is_Parallel
        assert z.is_Parallel
