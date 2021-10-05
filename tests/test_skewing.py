import pytest
import numpy as np

from conftest import assert_blocking, assert_structure
from devito.symbolics import MIN
from devito import (Grid, Eq, Function, TimeFunction, Operator,
                    solve, norm, Constant)
from devito.ir import Expression, Iteration, FindNodes


class TestCodeGen(object):

    '''
    Test code generation with blocking, skewing, wavefront and combinations thereof,
    tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected, norm_u, norm_v', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)',
         np.sqrt((14*18*17)*6**2 + (14*18*17)*5**2), 0]),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)',
         np.sqrt((14*18*17)*1**2 + (14*18*17)*1**2), 0]),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)',
         np.sqrt((14*18*17)*1**2 + (14*18*17)*1**2), 0]),
    ])
    def test_skewed_bounds(self, expr, expected, norm_u, norm_v):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(14, 18, 17))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        eqn = eval(expr)

        op = Operator(eqn, opt=('blocking', 'skewing'))
        op.apply(time_M=5)
        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1

        bns, _ = assert_blocking(op, {'x0_blk0'})

        iters = FindNodes(Iteration).visit(bns['x0_blk0'])
        assert len(iters) == 5
        assert iters[0].dim.parent is x
        assert iters[1].dim.parent is y
        assert iters[4].dim is z
        assert iters[2].dim.parent is iters[0].dim
        assert iters[3].dim.parent is iters[1].dim

        assert iters[0].symbolic_min == (iters[0].dim.parent.symbolic_min + time)
        assert iters[0].symbolic_max == (iters[0].dim.parent.symbolic_max + time)
        assert iters[1].symbolic_min == (iters[1].dim.parent.symbolic_min + time)
        assert iters[1].symbolic_max == (iters[1].dim.parent.symbolic_max + time)

        assert iters[2].symbolic_min == iters[0].dim
        assert iters[2].symbolic_max == MIN(iters[0].dim +
                                            iters[0].dim.symbolic_incr - 1,
                                            iters[0].dim.symbolic_max + time)
        assert iters[3].symbolic_min == iters[1].dim
        assert iters[3].symbolic_max == MIN(iters[1].dim +
                                            iters[1].dim.symbolic_incr - 1,
                                            iters[1].dim.symbolic_max + time)

        assert iters[4].symbolic_min == iters[4].dim.symbolic_min
        assert iters[4].symbolic_max == iters[4].dim.symbolic_max
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
    def test_no_sequential(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions

        u = Function(name='u', grid=grid)  # noqa
        v = Function(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', 'skewing'))
        op.apply()
        iters = FindNodes(Iteration).visit(op)
        assert len([i for i in iters if i.dim.is_Time]) == 0

        assert_blocking(op, {})  # no blocking is expected in the absence of time

        iters = FindNodes(Iteration).visit(op)
        assert_structure(op, ['x,y,z'])

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
    def test_skewing_codegen(self, expr, expected, skewing, blockinner):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)

        op = Operator(eqn, opt=('advanced', {'skewing': skewing, 'blocklevels': 0,
                                'blockinner': blockinner}))

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
            assert iters[3].symbolic_min == iters[3].dim.symbolic_min
            assert iters[3].symbolic_max == iters[3].dim.symbolic_max
        elif skewing and blockinner:
            assert iters[1].symbolic_min == (iters[1].dim.symbolic_min + time)
            assert iters[1].symbolic_max == (iters[1].dim.symbolic_max + time)
            assert iters[2].symbolic_min == (iters[2].dim.symbolic_min + time)
            assert iters[2].symbolic_max == (iters[2].dim.symbolic_max + time)
            assert iters[3].symbolic_min == (iters[3].dim.symbolic_min + time)
            assert iters[3].symbolic_max == (iters[3].dim.symbolic_max + time)
        elif not skewing and not blockinner:
            assert iters[1].symbolic_min == iters[1].dim.symbolic_min
            assert iters[1].symbolic_max == iters[1].dim.symbolic_max
            assert iters[2].symbolic_min == iters[2].dim.symbolic_min
            assert iters[2].symbolic_max == iters[2].dim.symbolic_max
            assert iters[3].symbolic_min == iters[3].dim.symbolic_min
            assert iters[3].symbolic_max == iters[3].dim.symbolic_max

        assert str(skewed[0]).replace(' ', '') == expected
        assert_structure(op, ['t,x,y,z'])

    '''
    Test code generation with wavefront
    '''
    @pytest.mark.parametrize('expr', ('Eq(u.forward, u + 1)',
                                      'Eq(u.forward, u.dx + u.dy + u.dz)',
                                      'Eq(u.forward, u.dx2 + u.dy2 + u.dz2)',
                                      'Eq(u.forward, u.dx4 + u.dy4 + u.dz4)',
                                      'Eq(u.forward, u.dx2 + u.dy2 + u.dz2)',
                                      'Eq(u.forward, u.dx2)', 'Eq(u.forward, u.dy2)',
                                      'Eq(u.forward, u.dz2)'))
    def test_sf_wavefront_codegen(self, expr):
        """Tests code generation with wavefront temporal blocking."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        so = 8
        u = TimeFunction(name='u', grid=grid, space_order = so)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)

        op = Operator(eqn, opt=('advanced', {'wavefront': True, 'blockinner': True}))
        op.apply(time_M=5)

        iters = FindNodes(Iteration).visit(op)

        assert len(iters) == 11
        assert_structure(op, ['time0_blk0x0_blk0y0_blk0z0_blk0tx0_blk1y0_blk1z0_blk1xyz'])

        assert iters[0].symbolic_min == iters[0].dim.symbolic_min
        assert iters[0].symbolic_max == 4*iters[0].dim.symbolic_max

        assert iters[1].symbolic_min == iters[1].dim.symbolic_min
        assert iters[1].symbolic_max == (iters[0].symbolic_max - 4*iters[0].symbolic_min +
                                         iters[1].dim.symbolic_max + 4)

        assert iters[2].symbolic_min == iters[2].dim.symbolic_min
        assert iters[2].symbolic_max == (iters[0].symbolic_max - 4*iters[0].symbolic_min +
                                         iters[2].dim.symbolic_max + 4)

        assert iters[3].symbolic_min == iters[3].dim.symbolic_min
        assert iters[3].symbolic_max == (iters[0].symbolic_max - 4*iters[0].symbolic_min +
                                         iters[3].dim.symbolic_max + 4)

        assert iters[4].symbolic_min == iters[4].dim.symbolic_min
        assert iters[4].symbolic_max == MIN(iters[4].dim.symbolic_max,
                                            iters[0].symbolic_max)


class TestWavefrontCorrectness(object):
    '''
    Test numerical corectness of operators with wavefronts/skewing
    '''
    def test_wave_correctness(self):

        nx = 19
        ny = 17
        nz = 25
        nt = 10
        nu = .5
        dx = 2. / (nx - 1)
        dy = 2. / (ny - 1)
        dz = 2. / (nz - 1)
        sigma = .25
        dt = sigma * dx * dz * dy / nu

        # Initialise u with hat function
        init_value = 6.5

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid, space_order=2)
        u.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        eq = Eq(u.dt, u.dx2 + u.dy2 + u.dz2)
        x, y, z = grid.dimensions
        stencil = solve(eq, u.forward)
        eq0 = Eq(u.forward, stencil)
        time_M = nt

        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eq0, opt=('advanced', {'openmp': True,
                      'wavefront': False, 'blocklevels': 1}))
        op.apply(time_M=time_M, dt=dt)
        norm_u = norm(u)
        u.data[:] = init_value

        op1 = Operator(eq0, opt=('advanced', {'skewing': True, 'openmp': True,
                                 'blocklevels': 1}))
        op1.apply(time_M=time_M, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)
        u.data[:] = init_value

        op2 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'wavefront': True, 'blocklevels': 2}))
        op2.apply(time_M=time_M, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)
        u.data[:] = init_value

        op3 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'wavefront': True, 'blocklevels': 3}))
        op3.apply(time_M=time_M, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)
        u.data[:] = init_value

        op4 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'wavefront': True, 'blocklevels': 4}))
        op4.apply(time_M=time_M, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)

        iters = FindNodes(Iteration).visit(op2)
        time_iter = [i for i in iters if i.dim.is_Time]

        assert len(time_iter) == 2
        assert_structure(op1, ['tx0_blk0y0_blk0xyz'])
        assert_structure(op2, ['time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1xyz'])
        assert_structure(op3, ['time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1x0_blk2'
                               'y0_blk2xyz'])
        assert_structure(op4, ['time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1x0_blk2y0_blk2'
                               'x0_blk3y0_blk3xyz'])

    def test_wave_correctness_II(self):
        nx = 29
        ny = 27
        nz = 48
        nt = 7
        nu = .5
        dx = 2. / (nx - 1)
        dy = 2. / (ny - 1)
        dz = 2. / (nz - 1)
        sigma = .25
        dt = sigma * dx * dz * dy / nu

        # Initialise u with hat function
        init_value = 1

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid, space_order=4, time_order=2)
        u.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        eq = Eq(u.dt, u.dx + u.dy + u.dz)
        x, y, z = grid.dimensions
        stencil = solve(eq, u.forward)
        eq0 = Eq(u.forward, stencil)
        time_M = nt

        op = Operator(eq0, opt=('advanced', {'openmp': True,
                                'wavefront': False, 'blocklevels': 2}))
        op.apply(time_M=time_M, dt=dt)
        norm_u = norm(u)
        u.data[:] = init_value

        op1 = Operator(eq0, opt=('advanced', {'skewing': True, 'openmp': True,
                                 'blocklevels': 1}))
        op1.apply(time_M=time_M, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-4, rtol=0)

        u.data[:] = init_value
        op2 = Operator(eq0, opt=('advanced', {'openmp': True, 'wavefront': True}))
        op2.apply(time_M=time_M, dt=dt)
        assert np.isclose(norm(u), norm_u, atol=1e-4, rtol=0)

        iters = FindNodes(Iteration).visit(op2)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 2
        assert_structure(op1, ['tx0_blk0y0_blk0xyz'])
        assert_structure(op2, ['time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1xyz'])

    @pytest.mark.parametrize('so', [4, 8, 16])
    @pytest.mark.parametrize('shape, nt',
                             [((2, 2, 2), 2), ((2, 2, 2), 1),  # Corner cases
                              ((14, 29, 16), 24), ((20, 22, 45), 19),
                              ((18, 18, 28), 26), ((18, 24, 34), 11),
                              ((19, 27, 24), 5), ((27, 25, 33), 19),
                              ((13, 28, 32), 21), ((33, 35, 32), 11)])
    def test_wave_correctness_III(self, so, shape, nt):
        nx, ny, nz = shape
        nu = .5
        dx = 2. / (nx - 1)
        dy = 2. / (ny - 1)
        dz = 2. / (nz - 1)
        sigma = .25
        dt = sigma * dx * dz * dy / nu

        # Initialise u
        init_value = 0.2

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid, space_order=so)
        u.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        a = Constant(name='a')
        eq = Eq(u.dt, a*u.laplace, subdomain=grid.interior)
        x, y, z = grid.dimensions
        stencil = solve(eq, u.forward)
        eq0 = Eq(u.forward, stencil)

        op = Operator(eq0, opt=('advanced', {'openmp': True,
                                'wavefront': False, 'blocklevels': 2}))

        op.apply(time_M=nt, dt=dt, a=nu)
        norm_u = norm(u)

        u.data[:] = init_value

        op1 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'wavefront': True, 'blocklevels': 2}))

        op1.apply(time_M=nt, dt=dt, a=nu)
        assert np.isclose(norm(u), norm_u, atol=1e-4, rtol=0)

        iters = FindNodes(Iteration).visit(op1)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 2
        assert_structure(op, ['tx0_blk0y0_blk0x0_blk1y0_blk1xyz'])
        assert_structure(op1, ['time0_blk0x0_blk0y0_blk0tx0_blk1y0_blk1xyz'])

    @pytest.mark.parametrize('so, shape, nt',
                             [(2, (15, 39, 21), 10), (2, (32, 16, 25), 13),
                              (4, (18, 38, 38), 66), (4, (28, 18, 14), 45),
                              (8, (17, 25, 42), 16), (8, (17, 25, 42), 17),
                              (16, (13, 38, 34), 21), (16, (16, 35, 32), 19)])
    def test_wave_correctness_IV(self, so, shape, nt):
        nx, ny, nz = shape
        nu = .5
        dx = 2. / (nx - 1)
        dy = 2. / (ny - 1)
        dz = 2. / (nz - 1)
        sigma = .25
        dt = sigma * dx * dz * dy / nu

        # Initialise u
        init_value = 0.1

        # Field initialization
        grid = Grid(shape=(nx, ny, nz))
        u = TimeFunction(name='u', grid=grid, space_order=so, time_order=2)
        u.data[:, :, :] = init_value

        # Create an equation with second-order derivatives
        a = Constant(name='a')
        eq = Eq(u.dt, a*u.laplace, subdomain=grid.interior)
        x, y, z = grid.dimensions
        stencil = solve(eq, u.forward)
        eq0 = Eq(u.forward, stencil)

        op = Operator(eq0, opt=('advanced', {'openmp': True,
                                'wavefront': False, 'blocklevels': 2}))

        op.apply(time_M=nt, dt=dt, a=nu)
        norm_u = norm(u)

        u.data[:] = init_value

        op2 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'wavefront': True, 'blocklevels': 2}))

        op2.apply(time_M=nt, dt=dt, a=nu)
        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)

    @pytest.mark.parametrize('nt', [2, 24, 8, 17, 16, 19, 41])
    @pytest.mark.parametrize('shape',
                             [((35, 39, 46)), ((62, 16, 45)),
                              ((38, 38, 38)), ((68, 20, 34)),
                              ((39, 37, 34)), ((17, 25, 42)),
                              ((33, 28, 24)), ((43, 35, 22))])
    def test_wave_correctness_V(self, shape, nt):
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
                                'wavefront': False, 'blocklevels': 2}))

        op.apply(time_M=nt)
        val0 = u.data[0, 1, 1, 1]
        val1 = u.data[1, 1, 1, 1]
        norm_u = norm(u)

        u.data[:, :, :] = init_value

        op2 = Operator(eq0, opt=('advanced', {'openmp': True,
                                              'wavefront': True, 'blocklevels': 2}))

        op2.apply(time_M=nt)
        assert np.all(u.data[0, :, :, :] == val0)
        assert np.all(u.data[1, :, :, :] == val1)
        assert np.isclose(norm(u), norm_u, atol=1e-3, rtol=0)

    @pytest.mark.parametrize('nt', [2, 9, 18])
    @pytest.mark.parametrize('shape',
                             [((25, 29, 26)), ((32, 16, 35)), ((33, 25, 12))])
    @pytest.mark.parametrize('so1', [2, 4, 8])
    @pytest.mark.parametrize('so2', [2, 4, 8])
    def test_wave_correctness_VI(self, shape, nt, so1, so2):
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

        op = Operator(eqns, opt=('advanced', {'openmp': True,
                                 'wavefront': False, 'blocklevels': 2}))

        op.apply(time_M=nt)
        norm_u = norm(u)

        u.data[:] = init_value

        op2 = Operator(eqns, opt=('advanced', {'openmp': True, 'wavefront': True}))
        op2.apply(time_M=nt)

        assert np.isclose(norm(u), norm_u, atol=1e-5, rtol=0)
