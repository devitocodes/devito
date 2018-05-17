import numpy as np

import pytest
from conftest import skipif_yask, configuration_override

from devito import (ConditionalDimension, Grid, Function, TimeFunction, Eq, Operator,  # noqa
                    Constant, SubDimension, DOMAIN, INTERIOR)
from devito.ir.iet import Iteration, FindNodes, retrieve_iteration_tree


@skipif_yask
class TestSubDimension(object):

    def test_domain_vs_interior(self):
        """
        Tests regions work properly in terms of code generation and runtime
        argument derivation. All regions but the default one, ``DOMAIN``,
        induce one or more :class:`SubDimension`s.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        u = TimeFunction(name='u', grid=grid)  # noqa
        eqs = [Eq(u.forward, u + 1),
               Eq(u.forward, u.forward + 2, region=INTERIOR)]

        op = Operator(eqs, dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2

        op.apply(time_M=1)
        assert np.all(u.data[1, 0, :, :] == 1)
        assert np.all(u.data[1, -1, :, :] == 1)
        assert np.all(u.data[1, :, 0, :] == 1)
        assert np.all(u.data[1, :, -1, :] == 1)
        assert np.all(u.data[1, :, :, 0] == 1)
        assert np.all(u.data[1, :, :, -1] == 1)
        assert np.all(u.data[1, 1:3, 1:3, 1:3] == 3)

    def test_subdim_middle(self):
        """
        Tests that instantiating SubDimensions using the classmethod
        constructors works correctly.
        """
        grid = Grid(shape=(4, 4, 4))
        x, y, z = grid.dimensions
        t = grid.stepping_dim  # noqa

        u = TimeFunction(name='u', grid=grid)  # noqa
        xi = SubDimension.middle(name='xi', parent=x,
                                 thickness_left=1,
                                 thickness_right=1)
        eqs = [Eq(u.forward, u + 1)]
        eqs = [e.subs(x, xi) for e in eqs]

        op = Operator(eqs)

        u.data[:] = 1.0
        op.apply(time_M=1)
        assert np.all(u.data[1, 0, :, :] == 1)
        assert np.all(u.data[1, -1, :, :] == 1)
        assert np.all(u.data[1, 1:3, :, :] == 2)

    def test_flow_detection_interior(self):
        """
        Test detection of flow directions when :class:`SubDimension`s are used
        (in this test they are induced by the ``INTERIOR`` region).

        Stencil uses values at new timestep as well as those at previous ones
        This forces an evaluation order onto x.
        Weights are:

               x=0     x=1     x=2     x=3
         t=N    2    ---3
                v   /
         t=N+1  o--+----4

        Flow dependency should traverse x in the negative direction

               x=2     x=3     x=4     x=5      x=6
        t=0             0   --- 0     -- 1    -- 0
                        v  /    v    /   v   /
        t=1            44 -+--- 11 -+--- 2--+ -- 0
        """

        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        u = TimeFunction(name='u', grid=grid, save=10, time_order=1, space_order=0)
        step = Eq(u.forward, 2*u
                  + 3*u.subs(x, x+x.spacing)
                  + 4*u.forward.subs(x, x+x.spacing),
                  region=INTERIOR)
        op = Operator(step)

        u.data[0, 5, 5] = 1.0
        op.apply(time_M=0)
        assert u.data[1, 5, 5] == 2
        assert u.data[1, 4, 5] == 11
        assert u.data[1, 3, 5] == 44
        assert u.data[1, 2, 5] == 4*44
        assert u.data[1, 1, 5] == 4*4*44

        # This point isn't updated because of the INTERIOR selection
        assert u.data[1, 0, 5] == 0

        assert np.all(u.data[1, 6:, :] == 0)
        assert np.all(u.data[1, :, 0:5] == 0)
        assert np.all(u.data[1, :, 6:] == 0)

    @pytest.mark.parametrize('exprs,expected,', [
        # Carried dependence in both /t/ and /x/
        (['Eq(u[t+1, x, y], u[t+1, x-1, y] + u[t, x, y], region=DOMAIN)'], 'y'),
        (['Eq(u[t+1, x, y], u[t+1, x-1, y] + u[t, x, y], region=INTERIOR)'], 'yi'),
        # Carried dependence in both /t/ and /y/
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], region=DOMAIN)'], 'x'),
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], region=INTERIOR)'], 'xi'),
        # Carried dependence in /y/, leading to separate /y/ loops, one
        # going forward, the other backward
        (['Eq(u[t+1, x, y], u[t+1, x, y-1] + u[t, x, y], region=INTERIOR)',
          'Eq(u[t+1, x, y], u[t+1, x, y+1] + u[t, x, y], region=INTERIOR)'], 'xi'),
    ])
    def test_iteration_property_parallel(self, exprs, expected):
        """Tests detection of sequental and parallel Iterations when applying
        equations over different regions."""
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions  # noqa
        t = grid.time_dim  # noqa
        u = TimeFunction(name='u', grid=grid, save=10, time_order=1)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs)
        iterations = FindNodes(Iteration).visit(op)
        assert all(i.is_Sequential for i in iterations if i.dim.name != expected)
        assert all(i.is_Parallel for i in iterations if i.dim.name == expected)

    @pytest.mark.parametrize('exprs,expected,', [
        (['Eq(u[t, x, yleft], u[t, x, yleft] + 1.)'], ['yleft']),
        # All outers are parallel, carried dependence in `yleft`, so no SIMD in `yleft`
        (['Eq(u[t, x, yleft], u[t, x, yleft+1] + 1.)'], []),
    ])
    def test_iteration_property_vector(self, exprs, expected):
        """Tests detection of vector Iterations when using subdimensions."""
        grid = Grid(shape=(20, 20))
        x, y = grid.dimensions  # noqa
        t = grid.time_dim  # noqa

        # The leftmost 10 elements
        yleft = SubDimension.left(name='yleft', parent=y, thickness=10) # noqa

        u = TimeFunction(name='u', grid=grid, save=10, time_order=0, space_order=1)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs)
        iterations = FindNodes(Iteration).visit(op)
        vectorizable = [i.dim.name for i in iterations if i.is_Vectorizable]
        assert set(vectorizable) == set(expected)


@skipif_yask
class TestConditionalDimension(object):

    """A collection of tests to check the correct functioning of
    :class:`ConditionalDimension`s."""

    def test_basic(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.), Eq(usave, u)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor)
                      for i in range((nt+factor-1)//factor)])

    # This test generates an openmp loop form which makes older gccs upset
    @configuration_override("openmp", False)
    def test_nothing_in_negative(self):
        """Test the case where when the condition is false, there is nothing to do."""
        nt = 4
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', save=nt, grid=grid)
        assert(grid.time_dim in u.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(usave, u)]
        op = Operator(eqns)

        u.data[:] = 1.0
        usave.data[:] = 0.0
        op.apply(time_m=1, time_M=1)
        assert np.allclose(usave.data, 0.0)

        op.apply(time_m=0, time_M=0)
        assert np.allclose(usave.data, 1.0)

    def test_laplace(self):
        grid = Grid(shape=(20, 20, 20))
        x, y, z = grid.dimensions
        time = grid.time_dim
        t = grid.stepping_dim
        tsave = ConditionalDimension(name='tsave', parent=time, factor=2)

        u = TimeFunction(name='u', grid=grid, save=None, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, time_dim=tsave,
                             time_order=0, space_order=0)

        steps = []
        # save of snapshot
        steps.append(Eq(usave, u))
        # standard laplace-like thing
        steps.append(Eq(u[t+1, x, y, z],
                        u[t, x, y, z] - u[t-1, x, y, z]
                        + u[t, x-1, y, z] + u[t, x+1, y, z]
                        + u[t, x, y-1, z] + u[t, x, y+1, z]
                        + u[t, x, y, z-1] + u[t, x, y, z+1]))

        op = Operator(steps)

        u.data[:] = 0.0
        u.data[0, 10, 10, 10] = 1.0
        op.apply(time_m=0, time_M=0)
        assert np.sum(u.data[0, :, :, :]) == 1.0
        assert np.sum(u.data[1, :, :, :]) == 7.0
        assert np.all(usave.data[0, :, :, :] == u.data[0, :, :, :])

    def test_as_expr(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=(nt+factor-1)//factor,
                             time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
                Eq(usave, time_subsampled * u)]
        op = Operator(eqns)
        op.apply(t=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        assert np.all([np.allclose(u2.data[i], i) for i in range(nt)])
        assert np.all([np.allclose(usave.data[i], i*factor*i)
                      for i in range((nt+factor-1)//factor)])

    def test_shifted(self):
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        u2 = TimeFunction(name='u2', grid=grid, save=nt)
        assert(time in u2.indices)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)
        usave = TimeFunction(name='usave', grid=grid, save=2, time_dim=time_subsampled)
        assert(time_subsampled in usave.indices)

        t_sub_shift = Constant(name='t_sub_shift', dtype=np.int32)

        eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2 + 1.),
                Eq(usave.subs(time_subsampled, time_subsampled - t_sub_shift), u)]
        op = Operator(eqns)

        # Starting at time_m=10, so time_subsampled - t_sub_shift is in range
        op.apply(time_m=10, time_M=nt-2, t_sub_shift=3)
        assert np.all(np.allclose(u.data[0], 8))
        assert np.all([np.allclose(u2.data[i], i - 10) for i in range(10, nt)])
        assert np.all([np.allclose(usave.data[i], 2+i*factor) for i in range(2)])

    def test_no_index(self):
        """Test behaviour when the ConditionalDimension is used as a symbol in
        an expression."""
        nt = 19
        grid = Grid(shape=(11, 11))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        assert(grid.stepping_dim in u.indices)

        v = Function(name='v', grid=grid)

        factor = 4
        time_subsampled = ConditionalDimension('t_sub', parent=time, factor=factor)

        eqns = [Eq(u.forward, u + 1), Eq(v, v + u*u*time_subsampled)]
        op = Operator(eqns)
        op.apply(t_M=nt-2)
        assert np.all(np.allclose(u.data[(nt-1) % 3], nt-1))
        # expected result is 1024
        # v = u[0]**2 * 0 + u[4]**2 * 1 + u[8]**2 * 2 + u[12]**2 * 3 + u[16]**2 * 4
        # with u[t] = t
        # v = 16 * 1 + 64 * 2 + 144 * 3 + 256 * 4 = 1600
        assert np.all(np.allclose(v.data, 1600))
