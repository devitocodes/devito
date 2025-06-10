import pytest
import numpy as np

from conftest import assert_blocking
from devito import Grid, Dimension, Eq, Function, TimeFunction, Operator, norm, Min # noqa
from devito.ir import Expression, Iteration, FindNodes


class TestCodeGenSkewing:

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
        op = Operator(eqn, opt=('blocking', {'skewing': True}))
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

        assert iters[2].symbolic_min == iters[2].dim.symbolic_min
        assert iters[2].symbolic_max == Min(iters[0].dim + iters[0].dim.symbolic_incr
                                            - 1, iters[0].dim.symbolic_max + time)
        assert iters[3].symbolic_min == iters[3].dim.symbolic_min
        assert iters[3].symbolic_max == Min(iters[1].dim + iters[1].dim.symbolic_incr
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
    def test_no_sequential(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        u = Function(name='u', grid=grid)  # noqa
        v = Function(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
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
        n = 4 if op._options['linearize'] else 0
        assert str(skewed[n]).replace(' ', '') == expected

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
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', {'blocklevels': 0, 'skewing': skewing,
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

        n = 6 if op._options['linearize'] else 0
        assert str(skewed[n]).replace(' ', '') == expected
