import pytest
import numpy as np

from conftest import assert_blocking, assert_structure, _R
from devito.symbolics import MIN
from devito import (Grid, Eq, Function, TimeFunction, Operator, norm,
                    Constant, solve)
from devito.ir import Expression, Iteration, FindNodes, FindSymbols


class TestCodeGenSkewing(object):

    '''
    Test code generation with blocking+skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected, norm_u, norm_v', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)',
         np.sqrt((16*16*16)*6**2 + (16*16*16)*5**2), 0]),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)',
         np.sqrt((16*16*16)*1**2 + (16*16*16)*1**2), 0]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)',
         np.sqrt((16*16*16)*1**2 + (16*16*16)*1**2), 0]),
    ])
    def test_skewed_bounds(self, expr, expected, norm_u, norm_v):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', 'skewing', {'blocktime': False}))
        op.apply(time_M=5)
        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1

        bns, _ = assert_blocking(op, {'x0_blk0'})

        iters = FindNodes(Iteration).visit(bns['x0_blk0'])
        assert len(iters) == 5
        assert iters[0].dim.parent is x
        assert iters[1].dim.parent is y
        assert iters[2].dim.parent is iters[0].dim
        assert iters[3].dim.parent is iters[1].dim
        assert iters[4].dim is z

        assert iters[0].symbolic_min == (iters[0].dim.parent.symbolic_min + time)
        assert iters[0].symbolic_max == (iters[0].dim.parent.symbolic_max + time)
        assert iters[1].symbolic_min == (iters[1].dim.parent.symbolic_min + time)
        assert iters[1].symbolic_max == (iters[1].dim.parent.symbolic_max + time)

        assert iters[2].symbolic_min == iters[2].dim.symbolic_min
        assert iters[2].symbolic_max == MIN(iters[0].dim + iters[0].dim.symbolic_incr
                                            - 1, iters[0].dim.symbolic_max + time)
        assert iters[3].symbolic_min == iters[3].dim.symbolic_min
        assert iters[3].symbolic_max == MIN(iters[1].dim + iters[1].dim.symbolic_incr
                                            - 1, iters[1].dim.symbolic_max + time)
        assert iters[4].symbolic_min == (iters[4].dim.symbolic_min)
        assert iters[4].symbolic_max == (iters[4].dim.symbolic_max)
        skewed = [i.expr for i in FindNodes(Expression).visit(bns['x0_blk0'])]
        assert str(skewed[0]).replace(' ', '') == expected
        assert np.isclose(norm(u), norm_u, rtol=1e-5)
        assert np.isclose(norm(v), norm_v, rtol=1e-5)

        u.data[:] = 0
        v.data[:] = 0
        op2 = Operator(eqn, opt=('advanced'))
        op2.apply(time_M=5)
        assert np.isclose(norm(u), norm_u, rtol=1e-5)
        assert np.isclose(norm(v), norm_v, rtol=1e-5)

    '''
    Test code generation with skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u, u + 1)',
          'Eq(u[x+1,y+1,z+1],u[x+1,y+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[x+1,y+1,z+1],v[x+1,y+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[x+1,y+1,z+1],v[x+1,y+1,z+1]+1)']),
    ])
    def test_notime_noskewing(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        u = Function(name='u', grid=grid)  # noqa
        v = Function(name='v', grid=grid)  # noqa

        eqn = eval(expr)
        op = Operator(eqn, opt=('blocking', {'skewing': True}))
        op.apply()
        iters = FindNodes(Iteration).visit(op)
        assert len([i for i in iters if i.dim.is_Time]) == 0

        assert_blocking(op, {})  # no blocking is expected in the absence of time

        iters = FindNodes(Iteration).visit(op)
        assert len(iters) == 3
        assert iters[0].dim is x
        assert iters[1].dim is y
        assert iters[2].dim is z

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]
        assert str(skewed[0]).replace(' ', '') == expected

    '''
    Test code generation with skewing only
    '''
    @pytest.mark.parametrize('expr, expected, skewing, blockinner', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)', True, False]),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)', True, False]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)', True, False]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x+1,y+1,z+1],v[t0,x+1,y+1,z+1]+1)', False, False]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z-time+1],v[t0,x-time+1,y-time+1,z-time+1]+1)',
          True, True]),
    ])
    def test_skewing_only(self, expr, expected, skewing, blockinner):
        """Tests skewing only with zero blocklevels and no timeblocking."""
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('advanced', {'blocklevels': 0, 'skewing': skewing,
                                'blockinner': blockinner, 'blocktime': False}))
        op.apply(time_M=5)

        iters = FindNodes(Iteration).visit(op)

        assert len(iters) == 4
        assert iters[0].dim is time
        assert iters[1].dim is x
        assert iters[2].dim is y
        assert iters[3].dim is z

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]

        if skewing and not blockinner:
            assert iters[1].symbolic_min == (iters[1].dim.symbolic_min + time)
            assert iters[1].symbolic_max == (iters[1].dim.symbolic_max + time)
            assert iters[2].symbolic_min == (iters[2].dim.symbolic_min + time)
            assert iters[2].symbolic_max == (iters[2].dim.symbolic_max + time)
            assert iters[3].symbolic_min == (iters[3].dim.symbolic_min)
            assert iters[3].symbolic_max == (iters[3].dim.symbolic_max)
        elif skewing and blockinner:
            assert iters[1].symbolic_min == iters[1].dim.symbolic_min + time
            assert iters[1].symbolic_max == iters[1].dim.symbolic_max + time
            assert iters[2].symbolic_min == iters[2].dim.symbolic_min + time
            assert iters[2].symbolic_max == iters[2].dim.symbolic_max + time
            assert iters[3].symbolic_min == iters[3].dim.symbolic_min + time
            assert iters[3].symbolic_max == iters[3].dim.symbolic_max + time
        elif not skewing and not blockinner:
            assert iters[1].symbolic_min == iters[1].dim.symbolic_min
            assert iters[1].symbolic_max == iters[1].dim.symbolic_max
            assert iters[2].symbolic_min == iters[2].dim.symbolic_min
            assert iters[2].symbolic_max == iters[2].dim.symbolic_max
            assert iters[3].symbolic_min == iters[3].dim.symbolic_min
            assert iters[3].symbolic_max == iters[3].dim.symbolic_max

        assert str(skewed[0]).replace(' ', '') == expected


class TestCodeGenTimeBlocking(object):
    '''
    Test time blocking code generation
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,-time+x+1,-time+y+1,z+1],u[t0,-time+x+1,-time+y+1,z+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,-time+x+1,-time+y+1,z+1],v[t0,-time+x+1,-time+y+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,-time+x+1,-time+y+1,z+1],v[t0,-time+x+1,-time+y+1,z+1]+1)'])
    ])
    def test_timeblocking(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('advanced', {'skewing': True}))
        op.apply(time_M=5)
        iters = FindNodes(Iteration).visit(op)

        assert_structure(op, ['time0_blk0x0_blk0y0_blk0txyz'])

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]

        assert (iters[0].symbolic_max == (iters[0].dim.symbolic_max))
        assert (iters[1].symbolic_max == (iters[1].dim.symbolic_max +
                iters[0].dim.symbolic_max - iters[0].dim.symbolic_min))

        assert (iters[2].symbolic_min == (iters[2].dim.symbolic_min))
        assert (iters[2].symbolic_max == (iters[2].dim.symbolic_max +
                iters[0].dim.symbolic_max - iters[0].dim.symbolic_min))

        assert str(skewed[0]).replace(' ', '') == expected

    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,-time+x+1,-time+y+1,z+1],u[t0,-time+x+1,-time+y+1,z+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,-time+x+1,-time+y+1,z+1],v[t0,-time+x+1,-time+y+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,-time+x+1,-time+y+1,z+1],v[t0,-time+x+1,-time+y+1,z+1]+1)'])
    ])
    def test_timeblocking_hierarchical(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('advanced', {'skewing': True, 'blocklevels': 2}))
        op.apply(time_M=5)
        iters = FindNodes(Iteration).visit(op)

        assert len(iters) == 9

        assert_structure(op, ['time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1xyz'])

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]

        assert iters[0].symbolic_max == iters[0].dim.symbolic_max
        assert iters[1].symbolic_max == (iters[1].dim.symbolic_max +
               iters[0].dim.symbolic_max - iters[0].dim.symbolic_min)

        assert iters[2].symbolic_min == iters[2].dim.symbolic_min
        assert iters[2].symbolic_max == (iters[2].dim.symbolic_max +
               iters[0].dim.symbolic_max - iters[0].dim.symbolic_min)

        assert str(skewed[0]).replace(' ', '') == expected


class TestWavefrontCorrectness(object):
    '''
    Test numerical corectness of operators with wavefronts/skewing
    '''
    @pytest.mark.parametrize('so', [2, 4, 8])
    @pytest.mark.parametrize('to', [1, 2])
    @pytest.mark.parametrize('shape, nt',
                             [((2, 2, 2), 2), ((2, 2, 2), 1),  # Corner cases
                              ((14, 29, 16), 24), ((20, 22, 45), 19),
                              ((13, 28, 32), 21), ((33, 35, 32), 11)])
    @pytest.mark.parametrize('time_iters, expected, skewing, bls', [
        ([1, 'tx0_blk0y0_blk0xyz', False, 1]),
        ([2, 'time0_blk0x0_blk0y0_blk0txyz', True, 1]),
        ([2, 'time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1x0_blk2y0_blk2xyz', True, 3]),
        ([2, 'time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1x0_blk2y0_blk2x0_blk3y0_blk3xyz',
          True, 4]),
    ])
    def test_wave_correctness(self, so, to, shape, nt, time_iters, expected, skewing,
                              bls):
        nx, ny, nz = shape
        nt = nt
        nu = .5
        dx = 2. / (nx - 1)
        dy = 2. / (ny - 1)
        dz = 2. / (nz - 1)
        sigma = .25
        dt = sigma * dx * dz * dy / nu

        # Initialise u
        init_value = 6.5

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid, space_order=so, time_order=to)
        u.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        a = Constant(name='a')
        eq = Eq(u.dt, a*u.laplace + 0.1, subdomain=grid.interior)
        stencil = solve(eq, u.forward)
        eq0 = Eq(u.forward, stencil)

        # List comprehension would need explicit locals/globals mappings to eval
        op0 = Operator(eq0, opt=('advanced'))
        op0.apply(time_M=nt, dt=dt)
        norm_u = norm(u)
        u.data[:] = init_value

        op1 = Operator(eq0, opt=('advanced', {'skewing': skewing,
                                 'blocklevels': bls}))
        op1.apply(time_M=nt, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-4, rtol=0)

        iters = FindNodes(Iteration).visit(op1)
        assert len([i for i in iters if i.dim.is_Time]) == time_iters
        assert_structure(op1, [expected])

    @pytest.mark.parametrize('nt', [4, 21, 17])
    @pytest.mark.parametrize('shape',
                             [((35, 39, 16)), ((19, 16, 45)), ((43, 15, 22))])
    def test_wave_correctness_II(self, shape, nt):
        '''
        Basic test
        '''
        nx, ny, nz = shape
        # Initialise u
        init_value = 0

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid)
        u.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        x, y, z = grid.dimensions
        eq0 = Eq(u.forward, u + 1.1)

        op = Operator(eq0, opt=('advanced', {'openmp': True,
                                'skewing': False, 'blocklevels': 2}))

        op.apply(time_M=nt)
        val0 = u.data[0, 1, 1, 1]
        val1 = u.data[1, 1, 1, 1]
        norm_u = norm(u)

        u.data[:, :, :] = init_value

        op2 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'skewing': True, 'blocklevels': 2}))

        op2.apply(time_M=nt)
        assert np.all(u.data[0, :, :, :] == val0)
        assert np.all(u.data[1, :, :, :] == val1)
        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)

    @pytest.mark.parametrize('nt', [2, 9])
    @pytest.mark.parametrize('shape',
                             [((25, 29, 26)), ((33, 25, 12))])
    @pytest.mark.parametrize('so1', [2, 4, 8])
    @pytest.mark.parametrize('so2', [2, 4, 8])
    def test_wave_correctness_coupled(self, shape, nt, so1, so2):
        "Test coupled equations"
        nx, ny, nz = shape

        # Initialise u
        init_value = 0

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid, space_order=so1)
        v = TimeFunction(name='v', grid=grid, space_order=so2)

        u.data[:, :, :] = init_value
        v.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        x, y, z = grid.dimensions
        eq0 = Eq(u.forward, u + u.backward + v)
        eq1 = Eq(v.forward, v + v.backward + u)
        eqns = [eq0, eq1]

        op0 = Operator(eqns, opt=('advanced', {'openmp': True,
                                  'skewing': False, 'blocklevels': 2}))
        op0.apply(time_M=nt)
        norm0 = norm(u)

        u.data[:] = init_value
        op1 = Operator(eqns, opt=('advanced', {'openmp': True, 'skewing': True}))
        op1.apply(time_M=nt)
        assert np.isclose(norm(u), norm0, atol=1e-5, rtol=0)


class TestEdgeCases(object):

    def test_full_shape_big_temporaries(self):
        """
        Test that if running with ``opt=advanced-fsg``, then the compiler uses
        temporaries spanning the whole grid rather than blocks.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        u.data_with_halo[:] = 0.5
        u1.data_with_halo[:] = 0.5

        # Leads to 3D aliases
        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z+1])*3. +
                               _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3. + 1.))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced-fsg', {'openmp': True, 'skewing': True,
                                 'cire-mingain': 0}))

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)