import pytest
import numpy as np
from math import floor

from sympy import sin, tan

from conftest import opts_tiling, assert_structure
from devito import (ConditionalDimension, Constant, Grid, Function, TimeFunction,
                    Eq, solve, Operator, SubDomain, SubDomainSet, Lt, SparseTimeFunction,
                    VectorFunction, TensorFunction)
from devito.ir import FindNodes, FindSymbols, Expression, Iteration, SymbolRegistry
from devito.tools import timed_region


class TestSubdomains:
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
            # _lower_exprs expects a SymbolRegistry, so create one
            expr = Operator._lower_exprs([eq0], options={},
                                         sregistry=SymbolRegistry())[0]
        assert str(expr.rhs) == 'ix*f[ix + 1, iy + 1] + iy'

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

    sd_specs = [('middle', 1, 7), ('middle', 2, 3), ('middle', 7, 1),
                ('middle', 5, 5), ('right', 3), ('right', 7), ('left', 3),
                ('left', 7)]

    @pytest.mark.parametrize('spec', sd_specs)
    @pytest.mark.parallel(mode=[2, 3])
    def test_subdomains_mpi(self, spec, mode):

        class sd0(SubDomain):
            name = 'd0'

            def __init__(self, spec):
                super().__init__()
                self.spec = spec

            def define(self, dimensions):
                x = dimensions[0]
                return {x: self.spec}
        s_d0 = sd0(spec)

        grid = Grid(shape=(11,), extent=(10.,), subdomains=(s_d0,))
        x = grid.dimensions[0]
        xd0 = grid.subdomains['d0'].dimensions[0]
        f = Function(name='f', grid=grid)

        Operator(Eq(f, f+1, subdomain=grid.subdomains['d0']))()

        # Sets ones on a global array according to the subdomains specified
        # then slices this according to the indices on this rank and compares
        # to the operator output.
        check = np.zeros(grid.shape)

        mM_map = {x.symbolic_min: 0, x.symbolic_max: grid.shape[0]-1}
        t_map = {tkn: tkn.value for tkn in xd0.thickness if tkn.value is not None}
        start = int(xd0.symbolic_min.subs({**mM_map, **t_map}))
        stop = int(xd0.symbolic_max.subs({**mM_map, **t_map})+1)

        check[slice(start, stop)] = 1

        assert np.all(check[grid.distributor.glb_slices[x]] == f.data)


class TestMultiSubDomain:

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
        assert len(reads) == 4  # f, g, h, mydomains

    def test_multi_eq_split(self):
        """
        Test cases where two loops over the same SubDomainSet will be
        separated by another loop.
        """
        # Note: a bug was found where this would cause SubDomainSet
        # bounds expressions not to be generated in the second loop over
        # the SubDomainSet
        class MSD(SubDomainSet):
            name = 'msd'

        msd = MSD(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(11, 11), subdomains=(msd,))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        eq0 = Eq(f, 1, subdomain=msd)
        eq1 = Eq(f, g)  # Dependency needed to fix equation order
        eq2 = Eq(g, 1, subdomain=msd)

        op = Operator([eq0, eq1, eq2])

        # Ensure the loop structure is correct
        # Note the two 'n0' correspond to the thickness definitions
        assert_structure(op,
                         ['n0', 'n0xy', 'xy', 'n0', 'n0xy'],
                         'n0xyxyn0xy')

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
    def test_subdomainset_mpi(self, mode):

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

        assert_structure(op, ['x,y', 't,n1', 't,n1,x,y'], 'x,y,t,n1,x,y')

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
                         ['x,y', 't,n1', 't,n1,x,y', 't,n2', 't,n2,x,y'],
                         'x,y,t,n1,x,y,n2,x,y')

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

        assert_structure(op, ['x,y', 't,n1', 't,n1,x,y',
                              't,n2', 't,n2,x,y', 't,n1', 't,n1,x,y'],
                         'x,y,t,n1,x,y,n2,x,y,n1,x,y')

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

        assert_structure(op, ['t,n1', 't,n1,x,y', 't,n1,x,y'],
                         't,n1,x,y,x,y')

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

        assert_structure(op, ['t', 't,n1', 't,n1,x,y', 't,n1', 't,n1,x,y'],
                         't,n1,x,y,n1,x,y')

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

        assert_structure(op, ['t,n0', 't,n0,ix0_blk0,iy0_blk0,x,y,z'],
                         't,n0,ix0_blk0,iy0_blk0,x,y,z')

        # Drag a rebuilt MultiSubDimension out of the operator
        dims = {d.name: d for d in FindSymbols('dimensions').visit(op)}
        xi = [d for d in dims['x']._defines if d.is_MultiSub]
        assert len(xi) == 1  # Sanity check
        xi = xi.pop()
        # Check that the correct number of thickness expressions are generated
        sdsexprs = [i.expr for i in FindNodes(Expression).visit(op)
                    if i.expr.rhs.is_Indexed
                    and i.expr.rhs.function is xi.functions]
        # The thickness expressions Eq(x_ltkn0, dummy[n0][0]), ...
        # should be scheduled once per dimension
        assert len(sdsexprs) == 6

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


class TestSubDomain_w_condition:

    def test_condition_w_subdomain_v0(self):

        shape = (10, )
        grid = Grid(shape=shape)
        x, = grid.dimensions

        class Middle(SubDomain):
            name = 'middle'

            def define(self, dimensions):
                return {x: ('middle', 2, 4)}

        mid = Middle()
        my_grid = Grid(shape=shape, subdomains=(mid, ))

        f = Function(name='f', grid=my_grid)

        sdf = Function(name='sdf', grid=my_grid)
        sdf.data[5:] = 1

        condition = Lt(sdf[mid.dimensions[0]], 1)

        ci = ConditionalDimension(name='ci', condition=condition,
                                  parent=mid.dimensions[0])

        op = Operator(Eq(f, f + 10, implicit_dims=ci,
                      subdomain=my_grid.subdomains['middle']))
        op.apply()

        assert_structure(op, ['x'], 'x')

    def test_condition_w_subdomain_v1(self):

        shape = (10, 10)
        grid = Grid(shape=shape)
        x, y = grid.dimensions

        class Middle(SubDomain):
            name = 'middle'

            def define(self, dimensions):
                return {x: x, y: ('middle', 2, 4)}

        mid = Middle()
        my_grid = Grid(shape=shape, subdomains=(mid, ))

        sdf = Function(name='sdf', grid=grid)
        sdf.data[:, 5:] = 1
        sdf.data[2:6, 3:5] = 1

        x1, y1 = mid.dimensions

        condition = Lt(sdf[x1, y1], 1)
        ci = ConditionalDimension(name='ci', condition=condition, parent=y1)

        f = Function(name='f', grid=my_grid)
        op = Operator(Eq(f, f + 10, implicit_dims=ci,
                      subdomain=my_grid.subdomains['middle']))

        op.apply()

        assert_structure(op, ['xy'], 'xy')

    def test_condition_w_subdomain_v2(self):

        shape = (10, 10)
        grid = Grid(shape=shape)
        x, y = grid.dimensions

        class Middle(SubDomain):
            name = 'middle'

            def define(self, dimensions):
                return {x: ('middle', 2, 4), y: ('middle', 2, 4)}

        mid = Middle()
        my_grid = Grid(shape=shape, subdomains=(mid, ))

        sdf = Function(name='sdf', grid=my_grid)
        sdf.data[2:4, 5:] = 1
        sdf.data[2:6, 3:5] = 1

        x1, y1 = mid.dimensions

        condition = Lt(sdf[x1, y1], 1)
        ci = ConditionalDimension(name='ci', condition=condition, parent=y1)

        f = Function(name='f', grid=my_grid)
        op = Operator(Eq(f, f + 10, implicit_dims=ci,
                      subdomain=my_grid.subdomains['middle']))

        op.apply()

        assert_structure(op, ['xy'], 'xy')


class TestRenaming:
    """
    Class for testing renaming of SubDimensions and MultiSubDimensions
    during compilation.
    """

    def test_subdimension_name_determinism(self):
        """
        Ensure that names allocated during compilation are deterministic in their
        ordering.
        """
        # Create two subdomains, two multisubdomains, then interleave them
        # across multiple equations

        class SD0(SubDomain):
            name = 'sd'

            def define(self, dimensions):
                x, y = dimensions
                return {x: x, y: ('right', 2)}

        class SD1(SubDomain):
            name = 'sd'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 1, 2), y: ('left', 2)}

        class MSD0(SubDomainSet):
            name = 'msd'

        class MSD1(SubDomainSet):
            name = 'msd'

        sd0 = SD0()
        sd1 = SD1()
        msd0 = MSD0(N=1, bounds=(1, 1, 1, 1))
        msd1 = MSD1(N=1, bounds=(1, 1, 1, 1))

        grid = Grid(shape=(11, 11), subdomains=(sd1, sd0, msd1, msd0))

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        eq0 = Eq(f, 1, subdomain=sd0)
        eq1 = Eq(g, f+1, subdomain=msd0)
        eq2 = Eq(h, f+g, subdomain=sd0)
        eq3 = Eq(g, h, subdomain=sd1)
        eq4 = Eq(f, f+1, subdomain=sd1)
        eq5 = Eq(f, h+1, subdomain=msd1)
        eq6 = Eq(f, g+1, subdomain=sd0)
        eq7 = Eq(g, 1, subdomain=msd1)
        eq8 = Eq(g, 1, subdomain=msd0)

        op = Operator([eq0, eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])
        assert_structure(op, ['xy', 'n0', 'n0xy', 'xy', 'xy',
                              'n1', 'n1xy', 'xy', 'n1', 'n1xy',
                              'n0', 'n0xy'],
                         'xyn0xyxyxyn1xyxyn1xyn0xy')


class ReducedDomain(SubDomain):
    name = 'reduced'

    def __init__(self, x_param, y_param, **kwargs):
        self._x_param = x_param
        self._y_param = y_param
        super().__init__(**kwargs)

    def define(self, dimensions):
        x, y = dimensions
        return {x: self._x_param if self._x_param is not None else x,
                y: self._y_param if self._y_param is not None else y}


class TestSubDomainFunctions:
    """Tests for functions defined on SubDomains"""

    _subdomain_specs = [('left', 3), ('right', 3), ('middle', 2, 3), None]

    @pytest.mark.parametrize('x', _subdomain_specs)
    @pytest.mark.parametrize('y', _subdomain_specs)
    @pytest.mark.parametrize('so', [2, 4])
    @pytest.mark.parametrize('functype', ['s', 'v', 't'])
    def test_function_data_shape(self, x, y, so, functype):
        """
        Check that defining a Function on a subset of a Grid results in arrays
        of the correct shape being allocated.
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        reduced_domain = ReducedDomain(x, y, grid=grid)
        if functype == 's':  # Scalar
            f = Function(name='f', grid=reduced_domain, space_order=so)
        elif functype == 'v':  # Vector
            f = VectorFunction(name='f', grid=reduced_domain, space_order=so)[0]
        else:  # Tensor
            f = TensorFunction(name='f', grid=reduced_domain, space_order=so)[0, 0]

        # Get thicknesses on each side
        def get_thickness(spec, shape):
            if spec is None:
                return 0, 0
            elif spec[0] == 'left':
                return 0, shape - spec[1]
            elif spec[0] == 'middle':
                return spec[1], spec[2]
            else:  # right
                return shape - spec[1], 0

        x_ltkn, x_rtkn = get_thickness(x, grid.shape[0])
        y_ltkn, y_rtkn = get_thickness(y, grid.shape[1])

        shape = (grid.shape[0] - x_ltkn - x_rtkn,
                 grid.shape[1] - y_ltkn - y_rtkn)

        assert f.dimensions == reduced_domain.dimensions
        assert f.data.shape == shape
        assert f.data_with_halo.shape == tuple(i+2*so for i in f.data.shape)
        assert f._distributor.shape == reduced_domain.shape
        for d in grid.dimensions:
            assert all([i == so for i in f._size_inhalo[d]])
            assert all([i == so for i in f._size_outhalo[d]])

    def test_slicing(self):
        """
        Test that slicing data for a Function defined on a SubDomain behaves
        as expected.
        """
        grid = Grid(shape=(10, 10), extent=(9., 9.))
        reduced_domain = ReducedDomain(('middle', 3, 1), ('right', 7), grid=grid)

        # 3 Functions for clarity and to minimise overlap
        f0 = Function(name='f0', grid=reduced_domain)
        f1 = Function(name='f1', grid=reduced_domain)
        f2 = Function(name='f2', grid=reduced_domain)

        # Check slicing
        f0.data[:] = 1
        f0.data[2:4, 1:-1] = 2
        f0.data[3:-2, 2:-3] = 3
        f0.data[-5:-3, -3:-2] = 4

        # Check slicing without ends and modulo slices
        f1.data[::2] = 1
        f1.data[::-2] = 2
        f1.data[2:] = 3
        f1.data[-2:] = 4

        # Check indexing of individual points
        f2.data[4, 2] = 5
        f2.data[0, 0] = 6
        f2.data[1, 1] = 7
        f2.data[0, -2] = 8
        f2.data[-2, 2] = 9

        check0 = np.full(f0.shape, 1.)
        check1 = np.zeros(f1.shape)
        check2 = np.zeros(f2.shape)

        check0[2:4, 1:-1] = 2
        check0[3:-2, 2:-3] = 3
        check0[-5:-3, -3:-2] = 4

        check1[::2] = 1
        check1[::-2] = 2
        check1[2:] = 3
        check1[-2:] = 4

        check2[4, 2] = 5
        check2[0, 0] = 6
        check2[1, 1] = 7
        check2[0, -2] = 8
        check2[-2, 2] = 9

        assert np.all(f0.data == check0)
        assert np.all(f1.data == check1)
        assert np.all(f2.data == check2)

    @pytest.mark.parametrize('x', _subdomain_specs)
    @pytest.mark.parametrize('y', _subdomain_specs)
    def test_basic_function(self, x, y):
        """
        Test a trivial operator with a single Function
        """

        grid = Grid(shape=(10, 10), extent=(9., 9.))
        reduced_domain = ReducedDomain(x, y, grid=grid)

        f = Function(name='f', grid=reduced_domain)
        eq = Eq(f, f+1)

        assert(f.shape == reduced_domain.shape)

        Operator(eq)()

        assert(np.all(f.data[:] == 1))

    def test_indices(self):
        """
        Test that indices when iterating over a Function defined on a
        SubDomain are aligned with the global indices
        """
        grid = Grid(shape=(10, 10), extent=(9., 9.))
        reduced_domain = ReducedDomain(('middle', 2, 3),
                                       ('right', 6),
                                       grid=grid)
        x, y = reduced_domain.dimensions
        f = Function(name='f', grid=reduced_domain)
        eq = Eq(f, x*y)

        Operator(eq)()

        check = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 8., 10., 12., 14., 16., 18., 0.],
                          [0., 12., 15., 18., 21., 24., 27., 0.],
                          [0., 16., 20., 24., 28., 32., 36., 0.],
                          [0., 20., 25., 30., 35., 40., 45., 0.],
                          [0., 24., 30., 36., 42., 48., 54., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0.]])
        assert np.all(f.data_with_halo == check)

    def test_mixed_functions(self):
        """
        Test with some Functions on a `SubDomain` and some not.
        """

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        grid = Grid(shape=(10, 10), extent=(9., 9.))
        mid = Middle(grid=grid)

        f = Function(name='f', grid=mid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)

        assert(f.shape == mid.shape)
        assert(g.shape == grid.shape)

        eq0 = Eq(f, g+f+1, subdomain=mid)
        eq1 = Eq(g, 2*f, subdomain=mid)
        eq2 = Eq(f, g+1, subdomain=mid)
        eq3 = Eq(h, g+1)

        op = Operator([eq0, eq1, eq2, eq3])

        assert_structure(op, ['x,y', 'x,y'], 'x,y,x,y')

        op()

        assert(np.all(f.data[:] == 3))
        assert(np.all(g.data[2:-2, 3:-1] == 2))

        h_check = np.full(grid.shape, 1)
        h_check[2:-2, 3:-1] = 3
        assert(np.all(h.data == h_check))

    @pytest.mark.parametrize('s_o', [2, 4, 6])
    def test_derivatives(self, s_o):
        """Test that derivatives are correctly evaluated."""

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        grid = Grid(shape=(10, 10), extent=(9., 9.))
        mid = Middle(grid=grid)

        f = Function(name='f', grid=mid, space_order=s_o)
        g = Function(name='g', grid=grid, space_order=s_o)

        fdx = Function(name='fdx', grid=mid)
        gdx = Function(name='gdx', grid=grid)

        fdy = Function(name='fdy', grid=mid)
        gdy = Function(name='gdy', grid=grid)

        msh_x, msh_y = np.meshgrid(np.arange(2, 8), np.arange(3, 9), indexing='ij')

        # One wavelength
        lam = 9./(2*np.pi)
        field = np.sin(lam*msh_x) + 0.4*np.sin(2*lam*msh_y) \
            + 0.2*np.sin(3*lam*msh_x + 2*lam*msh_y)

        f.data[:] = field
        g.data[2:-2, 3:-1] = field

        eq0 = Eq(fdx, f.dx, subdomain=mid)
        eq1 = Eq(fdy, f.dy, subdomain=mid)
        eq2 = Eq(gdx, g.dx, subdomain=mid)
        eq3 = Eq(gdy, g.dy, subdomain=mid)

        op = Operator([eq0, eq1, eq2, eq3])
        op()

        assert np.all(np.isclose(fdx.data[:], gdx.data[2:-2, 3:-1]))
        assert np.all(np.isclose(fdy.data[:], gdy.data[2:-2, 3:-1]))


class TestSubDomainFunctionsParallel:
    """Tests for functions defined on SubDomains with MPI"""
    # Note that some of the 'left' and 'right' SubDomains here are swapped
    # with 'middle' as they are local by default and so cannot be decomposed
    # across MPI ranks. Also more options here as there is a need to check
    # that empty MPI ranks don't cause issues
    _mpi_subdomain_specs = [('left', 3), ('middle', 0, 5),
                            ('right', 3), ('middle', 5, 0),
                            ('middle', 2, 3), ('middle', 1, 7),
                            None]

    @pytest.mark.parametrize('x', _mpi_subdomain_specs)
    @pytest.mark.parametrize('y', _mpi_subdomain_specs)
    @pytest.mark.parallel(mode=[(2, 'full')])  # Need to also test 3 and 4 in due course
    def test_function_data_shape_mpi(self, x, y, mode):
        """
        Check that defining a Function on a subset of a Grid results in arrays
        of the correct shape being allocated when decomposed with MPI.
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        reduced_domain = ReducedDomain(x, y, grid=grid)
        f = Function(name='f', grid=reduced_domain, space_order=2)

        g = Function(name='g', grid=grid, space_order=2)
        eq = Eq(g, g+1, subdomain=reduced_domain)
        Operator(eq)()

        slices = tuple(grid.distributor.glb_slices[dim]
                       for dim in grid.dimensions)

        assert np.count_nonzero(g.data) == f.data.size

        shape = []
        for i, s in zip(f._distributor.subdomain_interval, slices):
            if i is None:
                shape.append(s.stop - s.start)
            else:
                shape.append(max(0, 1 + min(s.stop-1, i.end) - max(s.start, i.start)))
        shape = tuple(shape)

        assert f.data.shape == shape
        assert f._distributor.parent.glb_shape == grid.shape
        assert f._distributor.glb_shape == reduced_domain.shape

    @pytest.mark.parametrize('x', _mpi_subdomain_specs)
    @pytest.mark.parametrize('y', _mpi_subdomain_specs)
    @pytest.mark.parallel(mode=[(2, 'full')])
    def test_basic_function_mpi(self, x, y, mode):
        """
        Test a trivial operator with a single Function
        """
        grid = Grid(shape=(11, 11), extent=(10., 10.))
        reduced_domain = ReducedDomain(x, y, grid=grid)

        f = Function(name='f', grid=reduced_domain)
        eq = Eq(f, f+1)

        Operator(eq)()

        assert(np.all(f.data == 1))

    @pytest.mark.parallel(mode=[(2, 'full')])
    def test_mixed_functions_mpi(self, mode):
        """
        Check that mixing Functions on SubDomains with regular Functions behaves
        correctly with MPI.
        """

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        grid = Grid(shape=(10, 10), extent=(9., 9.))
        mid = Middle(grid=grid)

        f = Function(name='f', grid=mid)
        g = Function(name='g', grid=grid)
        h = Function(name='h', grid=grid)
        i = Function(name='i', grid=mid)

        eq0 = Eq(f, g+f+1, subdomain=mid)
        eq1 = Eq(g, 2*f, subdomain=mid)
        eq2 = Eq(f, g+1, subdomain=mid)
        eq3 = Eq(h, g+1)
        eq4 = Eq(i, h+1, subdomain=mid)

        op = Operator([eq0, eq1, eq2, eq3, eq4])

        op()

        slices = tuple(grid._distributor.glb_slices[d]
                       for d in grid.dimensions)

        f_check = np.full(f.shape, 3, dtype=float)
        g_check = np.zeros(g.shape_global, dtype=float)
        g_check[2:-2, 3:-1] = 2
        h_check = np.full(h.shape_global, 1, dtype=float)
        h_check[2:-2, 3:-1] = 3
        i_check = np.full(i.shape, 4)

        assert np.all(f.data == f_check)
        assert np.all(g.data == g_check[slices])
        assert np.all(h.data == h_check[slices])
        assert np.all(i.data == i_check)

    def set_indices(data):
        """
        Set up individual indices for indexing/slicing check.
        """
        data[4, 2] = 1
        data[0, 0] = 2
        data[1, 1] = 3
        data[0, -2] = 4
        data[-2, 2] = 5

    def set_open_slices(data):
        """
        Set up open-ended slices for indexing/slicing check.
        """
        data[2:] = 1
        data[-2:] = 2

    def set_closed_slices(data):
        """
        Set up closed slices for indexing/slicing check.
        """
        data[:] = 1
        data[2:4, 1:-1] = 2
        data[3:-2, 2:-3] = 3
        data[-5:-3, -3:-2] = 4

    def set_modulo_slices(data):
        """
        Set up modulo slices for indexing/slices check.
        """
        # TODO: Assignment with negative modulo indexing currently doesn't work for
        # Functions defined on Grids or SubDomains. The two commented-out lines in
        # this function should be reinstated when this is fixed.
        data[::2, ::2] = 1
        # data[-2::-3] = 2
        # data[::-2] = 3
        data[:, ::3] = 4
        data[1::3] = 5

    _mpi_subdomain_specs_x = [('middle', 3, 1), None]
    _mpi_subdomain_specs_y = [('right', 7), ('middle', 2, 1), ('left', 7), None]

    @pytest.mark.parametrize('setter', [set_indices, set_open_slices,
                                        set_closed_slices, set_modulo_slices])
    @pytest.mark.parametrize('x', _mpi_subdomain_specs_x)
    @pytest.mark.parametrize('y', _mpi_subdomain_specs_y)
    @pytest.mark.parallel(mode=[(2, 'full')])
    def test_indexing_mpi(self, setter, x, y, mode):
        """
        Check that indexing into the Data of a Function defined on a SubDomain
        behaves as expected.
        """
        grid = Grid(shape=(10, 10), extent=(9., 9.))
        reduced_domain = ReducedDomain(x, y, grid=grid)

        f = Function(name='f', grid=reduced_domain)

        # Set the points
        setter(f.data)

        check = np.zeros(reduced_domain.shape)
        setter(check)

        # Can't gather inside the assert as it hangs due to the if condition
        data = f.data_gather()

        if f._distributor.myrank == 0:
            assert np.all(data == check)
        else:
            # Size zero array of None, so can't check "is None"
            # But check equal to None works, even though this is discouraged
            assert data == None  # noqa

    @pytest.mark.parametrize('s_o', [2, 4, 6])
    @pytest.mark.parallel(mode=[(2, 'full')])
    def test_derivatives(self, s_o, mode):
        """Test that derivatives are correctly evaluated."""

        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 3, 1)}

        grid = Grid(shape=(10, 10), extent=(9., 9.))
        mid = Middle(grid=grid)

        f = Function(name='f', grid=mid, space_order=s_o)
        g = Function(name='g', grid=grid, space_order=s_o)

        fdx = Function(name='fdx', grid=mid)
        gdx = Function(name='gdx', grid=grid)

        fdy = Function(name='fdy', grid=mid)
        gdy = Function(name='gdy', grid=grid)

        msh_x, msh_y = np.meshgrid(np.arange(2, 8), np.arange(3, 9), indexing='ij')

        # One wavelength
        lam = 9./(2*np.pi)
        field = np.sin(lam*msh_x) + 0.4*np.sin(2*lam*msh_y) \
            + 0.2*np.sin(3*lam*msh_x + 2*lam*msh_y)

        f.data[:] = field
        g.data[2:-2, 3:-1] = field

        eq0 = Eq(fdx, f.dx, subdomain=mid)
        eq1 = Eq(fdy, f.dy, subdomain=mid)
        eq2 = Eq(gdx, g.dx, subdomain=mid)
        eq3 = Eq(gdy, g.dy, subdomain=mid)

        op = Operator([eq0, eq1, eq2, eq3])
        op()

        assert np.all(np.isclose(fdx.data[:], gdx.data[2:-2, 3:-1]))
        assert np.all(np.isclose(fdy.data[:], gdy.data[2:-2, 3:-1]))

    @pytest.mark.parametrize('injection, norm', [(True, 15.834376),
                                                 (False, 1.0238341)])
    @pytest.mark.parallel(mode=[1, 2, 4])
    def test_diffusion(self, injection, norm, mode):
        """
        Test that a diffusion operator using Functions on SubDomains produces
        the same result as one on a Grid.
        """
        class Middle(SubDomain):

            name = 'middle'

            def define(self, dimensions):
                x, y = dimensions
                return {x: ('middle', 2, 2), y: ('middle', 2, 2)}

        dt = 0.1

        grid = Grid(shape=(14, 14), extent=(13., 13.))
        mid = Middle(grid=grid)

        f = TimeFunction(name='f', grid=mid, space_order=4)
        g = TimeFunction(name='g', grid=grid, space_order=4)

        pdef = f.dt - f.laplace
        pdeg = g.dt - g.laplace

        eqf = Eq(f.forward, solve(pdef, f.forward), subdomain=mid)
        eqg = Eq(g.forward, solve(pdeg, g.forward), subdomain=mid)

        if injection:
            srcf = SparseTimeFunction(name='srcf', grid=grid, npoint=1, nt=10)
            srcg = SparseTimeFunction(name='srcg', grid=grid, npoint=1, nt=10)

            srcf.coordinates.data[:] = 6.5
            srcg.coordinates.data[:] = 6.5

            srcf.data[:, 0] = np.arange(10)
            srcg.data[:, 0] = np.arange(10)

            sf = srcf.inject(field=f.forward, expr=srcf)
            sg = srcg.inject(field=g.forward, expr=srcg)

            Operator([eqf] + sf)(dt=dt)
            Operator([eqg] + sg)(dt=dt)
        else:
            f.data[:, 4:-4, 4:-4] = 1
            g.data[:, 6:-6, 6:-6] = 1

            Operator(eqf)(t_M=10, dt=dt)
            Operator(eqg)(t_M=10, dt=dt)

        fdata = f.data_gather()
        gdata = g.data_gather()

        if grid.distributor.myrank == 0:
            assert np.all(np.isclose(fdata[:], gdata[:, 2:-2, 2:-2]))
            assert np.isclose(np.linalg.norm(fdata[:]), norm)
