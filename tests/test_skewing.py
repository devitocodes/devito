import pytest

from sympy import Add, cos, sin, sqrt  # noqa

from devito.core.autotuning import options  # noqa
from devito import Function, TimeFunction, Grid, Operator, Eq # noqa
from devito.ir import Expression, Iteration, FindNodes


class TestCodeGenSkew(object):

    '''
    Test code generation with blocking+skewing, tests adapted from test_operator.py
    '''
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)']),
    ])
    def test_skewed_bounds(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('blocking', 'skewing'))

        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 1

        for i in ['bf0']:
            assert i in op._func_table
            iters = FindNodes(Iteration).visit(op._func_table[i].root)
            assert len(iters) == 5
            assert iters[0].dim.parent is x
            assert iters[1].dim.parent is y
            assert iters[4].dim is z
            assert iters[2].dim.parent is iters[0].dim
            assert iters[3].dim.parent is iters[1].dim

            assert (iters[2].symbolic_min == (iters[0].dim + time))
            assert (iters[2].symbolic_max == (iters[0].dim + time +
                                              iters[0].dim.symbolic_incr - 1))
            assert (iters[3].symbolic_min == (iters[1].dim + time))
            assert (iters[3].symbolic_max == (iters[1].dim + time +
                                              iters[1].dim.symbolic_incr - 1))

            assert (iters[4].symbolic_min == (iters[4].dim.symbolic_min))
            assert (iters[4].symbolic_max == (iters[4].dim.symbolic_max))
            skewed = [i.expr for i in FindNodes(Expression).visit(op._func_table[i].root)]
            assert str(skewed[0]).replace(' ', '') == expected

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

        iters = FindNodes(Iteration).visit(op)
        time_iter = [i for i in iters if i.dim.is_Time]
        assert len(time_iter) == 0

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
    @pytest.mark.parametrize('expr, expected', [
        (['Eq(u.forward, u + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],u[t0,x-time+1,y-time+1,z+1]+1)']),
        (['Eq(u.forward, v + 1)',
          'Eq(u[t1,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)']),
        (['Eq(u, v + 1)',
          'Eq(u[t0,x-time+1,y-time+1,z+1],v[t0,x-time+1,y-time+1,z+1]+1)']),
    ])
    def test_skewing_codegen(self, expr, expected):
        """Tests code generation on skewed indices."""
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid)  # noqa
        v = TimeFunction(name='v', grid=grid)  # noqa
        eqn = eval(expr)
        # List comprehension would need explicit locals/globals mappings to eval
        op = Operator(eqn, opt=('skewing'))

        iters = FindNodes(Iteration).visit(op)

        assert len(iters) == 4
        assert iters[0].dim is time
        assert iters[1].dim is x
        assert iters[2].dim is y
        assert iters[3].dim is z

        skewed = [i.expr for i in FindNodes(Expression).visit(op)]

        for iter in iters[1:2]:
            assert (iter.symbolic_min == (iter.dim.symbolic_min + time))
            assert (iter.symbolic_max == (iter.dim.symbolic_max + time))

        assert str(skewed[0]).replace(' ', '') == expected
