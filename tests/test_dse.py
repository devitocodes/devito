from sympy import Add
import numpy as np
import pytest
from cached_property import cached_property

from conftest import skipif, EVAL, _R, assert_structure, assert_blocking  # noqa
from devito import (NODE, Eq, Inc, Constant, Function, TimeFunction, SparseTimeFunction,  # noqa
                    Dimension, SubDimension, ConditionalDimension, DefaultDimension, Grid,
                    Operator, norm, grad, div, dimensions, switchconfig, configuration,
                    centered, first_derivative, solve, transpose, cos, sin, sqrt)
from devito.exceptions import InvalidArgument, InvalidOperator
from devito.finite_differences.differentiable import diffify
from devito.ir import (Conditional, DummyEq, Expression, Iteration, FindNodes,
                       FindSymbols, ParallelIteration, retrieve_iteration_tree)
from devito.passes.clusters.aliases import collect
from devito.passes.clusters.cse import Temp, _cse
from devito.passes.iet.parpragma import VExpanded
from devito.symbolics import estimate_cost, pow_to_mul, indexify
from devito.tools import as_tuple, generator
from devito.types import Scalar, Array

from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, AcquisitionGeometry
from examples.seismic.tti import AnisotropicWaveSolver


def test_scheduling_after_rewrite():
    """Tests loop scheduling after expression hoisting."""
    grid = Grid((10, 10))
    u1 = TimeFunction(name="u1", grid=grid, save=10, time_order=2)
    u2 = TimeFunction(name="u2", grid=grid, time_order=2)
    sf1 = SparseTimeFunction(name='sf1', grid=grid, npoint=1, nt=10)
    const = Function(name="const", grid=grid, space_order=2)

    # Deliberately inject into u1, rather than u1.forward, to create a WAR
    eqn1 = Eq(u1.forward, u1 + sin(const))
    eqn2 = sf1.inject(u1.forward, expr=sf1)
    eqn3 = Eq(u2.forward, u2 - u1.dt2 + sin(const))

    op = Operator([eqn1] + eqn2 + [eqn3])
    trees = retrieve_iteration_tree(op)

    # Check loop nest structure
    assert all(i.dim is j for i, j in zip(trees[0], grid.dimensions))  # time invariant
    assert trees[1].root.dim is grid.time_dim
    assert all(trees[1].root.dim is tree.root.dim for tree in trees[1:])


@pytest.mark.parametrize('exprs,expected', [
    # Simple cases
    (['Eq(tu, 2/(t0 + t1))', 'Eq(ti0, t0 + t1)', 'Eq(ti1, t0 + t1)'],
     ['t0 + t1', '2/r0', 'r0', 'r0']),
    (['Eq(tu, 2/(t0 + t1))', 'Eq(ti0, 2/(t0 + t1) + 1)', 'Eq(ti1, 2/(t0 + t1) + 1)'],
     ['1/(t0 + t1)', '2*r5', 'r3 + 1', 'r3', 'r2', 'r2']),
    (['Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     ['ti0[x, y, z] + ti1[x, y, z]',
      'r0*(t0 + t1) + r0*(tv[t, x, y, z] + tw[t, x, y, z] + 5.0)']),
    (['Eq(tu, t0/t1)', 'Eq(ti0, 2 + t0/t1)', 'Eq(ti1, 2 + t0/t1)'],
     ['t0/t1', 'r2 + 2', 'r2', 'r1', 'r1']),
    # Across expressions
    (['Eq(tu, tv*4 + tw*5 + tw*5*t0)', 'Eq(tv, tw*5)'],
     ['5*tw[t, x, y, z]', 'r0 + 5*t0*tw[t, x, y, z] + 4*tv[t, x, y, z]', 'r0']),
    # Intersecting
    pytest.param(['Eq(tu, ti0*ti1 + ti0*ti1*t0 + ti0*ti1*t0*t1)'],
                 ['ti0*ti1', 'r0', 'r0*t0', 'r0*t0*t1'],
                 marks=pytest.mark.xfail),
    # Divisions (== powers with negative exponenet) are always captured
    (['Eq(tu, tv**-1*(tw*5 + tw*5*t0))', 'Eq(ti0, tv**-1*t0)'],
     ['1/tv[t, x, y, z]', 'r0*(5*t0*tw[t, x, y, z] + 5*tw[t, x, y, z])', 'r0*t0']),
    # `compact_temporaries` must detect chains of isolated temporaries
    (['Eq(t0, tv)', 'Eq(t1, t0)', 'Eq(t2, t1)', 'Eq(tu, t2)'],
     ['tv[t, x, y, z]']),
    # Dimension-independent data dependences should be a stopper for CSE
    (['Eq(tu.forward, tu.dx + 1)', 'Eq(tv.forward, tv.dx + 1)',
      'Eq(tw.forward, tv.dt + 1)', 'Eq(tz.forward, tv.dt + 2)'],
     ['1/h_x', '1/dt', '-r1*tv[t, x, y, z]',
      '-r2*tu[t, x, y, z] + r2*tu[t, x + 1, y, z] + 1',
      '-r2*tv[t, x, y, z] + r2*tv[t, x + 1, y, z] + 1',
      'r0 + r1*tv[t + 1, x, y, z] + 1',
      'r0 + r1*tv[t + 1, x, y, z] + 2']),
    # Fancy use case with lots of temporaries
    (['Eq(tu.forward, tu.dx + 1)', 'Eq(tv.forward, tv.dx + 1)',
      'Eq(tw.forward, tv.dt.dx2.dy2 + 1)', 'Eq(tz.forward, tv.dt.dy2.dx2 + 2)'],
     ['1/h_x', 'h_x**(-2)', 'h_y**(-2)', '1/dt', '-r11*tv[t, x, y, z]', '-2.0*r12',
      '-2.0*r13', '-r11*tv[t, x + 1, y, z] + r11*tv[t + 1, x + 1, y, z]',
      '-r11*tv[t, x - 1, y, z] + r11*tv[t + 1, x - 1, y, z]',
      '-r11*tv[t, x, y + 1, z] + r11*tv[t + 1, x, y + 1, z]',
      '-r11*tv[t, x + 1, y + 1, z] + r11*tv[t + 1, x + 1, y + 1, z]',
      '-r11*tv[t, x - 1, y + 1, z] + r11*tv[t + 1, x - 1, y + 1, z]',
      '-r11*tv[t, x, y - 1, z] + r11*tv[t + 1, x, y - 1, z]',
      '-r11*tv[t, x + 1, y - 1, z] + r11*tv[t + 1, x + 1, y - 1, z]',
      '-r11*tv[t, x - 1, y - 1, z] + r11*tv[t + 1, x - 1, y - 1, z]',
      '-r14*tu[t, x, y, z] + r14*tu[t, x + 1, y, z] + 1',
      '-r14*tv[t, x, y, z] + r14*tv[t, x + 1, y, z] + 1',
      'r12*(r0*r13 + r1*r13 + r2*r8) + r12*(r13*r3 + r13*r4 + r5*r8) + '
      'r9*(r13*r6 + r13*r7 + r8*(r10 + r11*tv[t + 1, x, y, z])) + 1',
      'r13*(r0*r12 + r12*r3 + r6*r9) + r13*(r1*r12 + r12*r4 + r7*r9) + '
      'r8*(r12*r2 + r12*r5 + r9*(r10 + r11*tv[t + 1, x, y, z])) + 2']),
])
def test_cse(exprs, expected):
    """Test common subexpressions elimination."""
    grid = Grid((3, 3, 3))
    dims = grid.dimensions

    tu = TimeFunction(name="tu", grid=grid, space_order=2)  # noqa
    tv = TimeFunction(name="tv", grid=grid, space_order=2)  # noqa
    tw = TimeFunction(name="tw", grid=grid, space_order=2)  # noqa
    tz = TimeFunction(name="tz", grid=grid, space_order=2)  # noqa
    ti0 = Array(name='ti0', shape=(3, 5, 7), dimensions=dims).indexify()  # noqa
    ti1 = Array(name='ti1', shape=(3, 5, 7), dimensions=dims).indexify()  # noqa
    t0 = Temp(name='t0')  # noqa
    t1 = Temp(name='t1')  # noqa
    t2 = Temp(name='t2')  # noqa

    # List comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = DummyEq(indexify(diffify(eval(e).evaluate)))

    counter = generator()
    make = lambda: Temp(name='r%d' % counter()).indexify()
    processed = _cse(exprs, make)
    assert len(processed) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(processed, expected))


@pytest.mark.parametrize('expr,expected', [
    ('2*fa[x] + fb[x]', '2*fa[x] + fb[x]'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('fa[x]**2 + fb[x]**3', 'fa[x]*fa[x] + fb[x]*fb[x]*fb[x]'),
    ('3*fa[x]**4', '3*(fa[x]*fa[x]*fa[x]*fa[x])'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('1/(fa[x]**2)', '1/(fa[x]*fa[x])'),
    ('1/(fb[x]**2 + 1)', '1/(fb[x]*fb[x] + 1)'),
    ('1/(fa[x] + fb[x])', '1/(fa[x] + fb[x])'),
    ('3*sin(fa[x])**2', '3*(sin(fa[x])*sin(fa[x]))'),
    ('fa[x]/(fb[x]**2)', 'fa[x]/((fb[x]*fb[x]))'),
    ('(fa[x]**0.5)**2', 'fa[x]'),
])
def test_pow_to_mul(expr, expected):
    grid = Grid((4, 5))
    x, y = grid.dimensions

    s = Scalar(name='s')  # noqa
    fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
    fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa

    assert str(pow_to_mul(eval(expr))) == expected


@pytest.mark.parametrize('expr,expected,estimate', [
    ('Eq(t0, t1)', 0, False),
    ('Eq(t0, -t1)', 0, False),
    ('Eq(t0, -t1)', 0, True),
    ('Eq(t0, fa[x] + fb[x])', 1, False),
    ('Eq(t0, fa[x + 1] + fb[x - 1])', 1, False),
    ('Eq(t0, fa[fb[x+1]] + fa[x])', 1, False),
    ('Eq(t0, fa[fb[x+1]] + fc[x+2, y+1])', 1, False),
    ('Eq(t0, t1*t2)', 1, False),
    ('Eq(t0, 2.*t0*t1*t2)', 3, False),
    ('Eq(t0, cos(t1*t2))', 2, False),
    ('Eq(t0, (t1*t2)**0)', 0, False),
    ('Eq(t0, (t1*t2)**t1)', 2, False),
    ('Eq(t0, (t1*t2)**2)', 3, False),  # SymPy distributes integer exponents in a Mul
    ('Eq(t0, 2.*t0*t1*t2 + t0*fa[x+1])', 5, False),
    ('Eq(t0, (2.*t0*t1*t2 + t0*fa[x+1])*3. - t0)', 7, False),
    ('[Eq(t0, (2.*t0*t1*t2 + t0*fa[x+1])*3. - t0), Eq(t0, cos(t1*t2))]', 9, False),
    ('Eq(t0, cos(fa*fb))', 101, True),
    ('Eq(t0, cos(fa[x]*fb[x]))', 101, True),
    ('Eq(t0, cos(t1*t2))', 101, True),
    ('Eq(t0, cos(c*c))', 101, True),
    ('Eq(t0, t1**3)', 2, True),
    ('Eq(t0, t1**4)', 3, True),
    ('Eq(t0, t2*t1**-1)', 6, True),
    ('Eq(t0, t1**t2)', 50, True),
    ('Eq(t0, 3.2/h_x)', 6, True),  # seen as `3.2*(1/h_x)`, so counts as 2
    ('Eq(t0, 3.2/h_x*fa + 2.4/h_x*fb)', 15, True),  # `pow(...constants...)` counts as 1
])
def test_estimate_cost(expr, expected, estimate):
    # Note: integer arithmetic isn't counted
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions  # noqa

    h_x = x.spacing  # noqa
    c = Constant(name='c')  # noqa
    t0 = Scalar(name='t0')  # noqa
    t1 = Scalar(name='t1')  # noqa
    t2 = Scalar(name='t2')  # noqa
    fa = Function(name='fa', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fb = Function(name='fb', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fc = Function(name='fc', grid=grid)  # noqa

    assert estimate_cost(eval(expr), estimate) == expected


@pytest.mark.parametrize('exprs,exp_u,exp_v', [
    (['Eq(s, 0, implicit_dims=(x, y))', 'Eq(s, s + 4, implicit_dims=(x, y))',
      'Eq(u, s)'], 4, 0),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Eq(s, s + s + 4, implicit_dims=(x, y))',
      'Eq(s, s + 4, implicit_dims=(x, y))', 'Eq(u, s)'], 8, 0),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))',
      'Eq(u, s)'], 4, 0),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))', 'Eq(v, s)',
      'Eq(u, s)'], 4, 4),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))', 'Eq(v, s)',
      'Eq(s, s + 4, implicit_dims=(x, y))', 'Eq(u, s)'], 8, 4),
    (['Eq(s, 0, implicit_dims=(x, y))', 'Inc(s, 4, implicit_dims=(x, y))', 'Eq(v, s)',
      'Inc(s, 4, implicit_dims=(x, y))', 'Eq(u, s)'], 8, 4),
    (['Eq(u, 0)', 'Inc(u, 4)', 'Eq(v, u)', 'Inc(u, 4)'], 8, 4),
    (['Eq(u, 1)', 'Eq(v, 4)', 'Inc(u, v)', 'Inc(v, u)'], 5, 9),
])
def test_makeit_ssa(exprs, exp_u, exp_v):
    """
    A test building Operators with non-trivial sequences of input expressions
    that push hard on the `makeit_ssa` utility function.
    """
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions  # noqa
    u = Function(name='u', grid=grid)  # noqa
    v = Function(name='v', grid=grid)  # noqa
    s = Scalar(name='s')  # noqa

    # List comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)

    op = Operator(exprs)
    op.apply()

    assert np.all(u.data == exp_u)
    assert np.all(v.data == exp_v)


@pytest.mark.parametrize('opt', ['noop', 'advanced'])
def test_time_dependent_split(opt):
    grid = Grid(shape=(10, 10))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=3)
    v = TimeFunction(name='v', grid=grid, time_order=2, space_order=0, save=3)

    # The second equation needs a full loop over x/y for u then
    # a full one over x.y for v
    eq = [Eq(u.forward, 2 + grid.time_dim),
          Eq(v.forward, u.forward.dx + u.forward.dy + 1)]
    op = Operator(eq, opt=opt)

    trees = retrieve_iteration_tree(op)
    assert len(trees) == 2

    op()

    assert np.allclose(u.data[2, :, :], 3.0)
    assert np.allclose(v.data[1, 1:-1, 1:-1], 1.0)


class TestLifting(object):

    @pytest.mark.parametrize('exprs,expected', [
        # none (different distance)
        (['Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
         'Inc(h1[0, 0], 1, implicit_dims=(t, x, y))'],
         [6., 0., 0.]),
        (['Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
         'Eq(h1[0, 0], y, implicit_dims=(t, x, y))'],
         [2., 0., 0.]),
        (['Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
         'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         [0., 1., 2.]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
         'Eq(h1[0, y], 3 - y, implicit_dims=(t, x, y))'],
         [3., 2., 1.]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
          'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         [0., 1., 2.]),
        (['Eq(y.symbolic_min, g[0, 0], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, 2], implicit_dims=(t, x))',
          'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         [0., 1., 2.]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, y], y, implicit_dims=(t, x, y))'],
         [0., 2., 6.]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, x], y, implicit_dims=(t, x, y))'],
         [3., 3., 2.]),
        (['Eq(y.symbolic_min, g[0, 0], implicit_dims=(t, x))',
          'Inc(h1[0, y], x, implicit_dims=(t, x, y))'],
         [3., 3., 3.]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, x], y.symbolic_min, implicit_dims=(t, x))'],
         [2., 2., 2.]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Inc(h1[0, x], y.symbolic_min, implicit_dims=(t, x, y))'],
         [2., 2., 2.]),
        (['Eq(y.symbolic_min, g[0, 2], implicit_dims=(t, x))',
          'Eq(h1[0, x], y.symbolic_min, implicit_dims=(t, x))'],
         [2., 2., 2.]),
        (['Eq(y.symbolic_min, g[0, x], implicit_dims=(t, x))',
          'Eq(y.symbolic_max, g[0, x]-1, implicit_dims=(t, x))',
          'Eq(h1[0, y], y, implicit_dims=(t, x, y))'],
         [0., 0., 0.])
    ])
    def test_edge_cases(self, exprs, expected):
        t, x, y = dimensions('t x y')

        g = TimeFunction(name='g', shape=(1, 3), dimensions=(t, x),
                         time_order=0, dtype=np.int32)
        g.data[0, :] = [0, 1, 2]
        h1 = TimeFunction(name='h1', shape=(1, 3), dimensions=(t, y), time_order=0)
        h1.data[0, :] = 0

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs)
        op.apply()

        assert np.all(h1.data == expected)

    @pytest.mark.parametrize('exprs,expected,visit', [
        (['Eq(f, f + g*2, implicit_dims=(time, x, y))',
          'Eq(u, (f + f[y+1])*g)'],
         ['txy', 'txy'], 'txyy'),
    ])
    def test_contracted(self, exprs, expected, visit):
        """
        Test that in situations such as

            for i
              for x
                r = f(a[x])

        the `r` statement isn't lifted outside of `i`, since we're not recording
        each of the computed `x` value (IOW, we're writing to `r` rather than `r[x]`).
        """
        grid = Grid(shape=(3, 3), dtype=np.int32)
        x, y = grid.dimensions
        time = grid.time_dim  # noqa

        f = Function(name='f', grid=grid, shape=(3,), dimensions=(y,))  # noqa
        g = Function(name='g', grid=grid)  # noqa
        u = TimeFunction(name='u', grid=grid, time_order=0)  # noqa

        # List comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = eval(e)

        op = Operator(exprs)

        trees = retrieve_iteration_tree(op)
        iters = FindNodes(Iteration).visit(op)
        assert len(trees) == len(expected)
        # mapper just makes it quicker to write out the test parametrization
        mapper = {'time': 't'}
        assert ["".join(mapper.get(i.dim.name, i.dim.name) for i in j)
                for j in trees] == expected
        assert "".join(mapper.get(i.dim.name, i.dim.name) for i in iters) == visit


class TestAliases(object):

    @pytest.mark.parametrize('exprs,expected', [
        # none (different distance)
        (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x])'],
         []),
        # none (different dimension)
        (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] + fb[y])'],
         []),
        # none (different operation)
        (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] - fb[x])'],
         []),
        # simple
        (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x+1])',
          'Eq(t2, fa[x+2] + fb[x+2])'],
         ['fa[x+1] + fb[x+1]']),
        # 2D simple
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(t1, fc[x+1,y+1] + fd[x+1,y+1])'],
         ['fc[x+1,y+1] + fd[x+1,y+1]']),
        # 2D with stride
        (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])'],
         ['fc[x+1,y+1] + fd[x+2,y+3]']),
        # 2D with subdimensions
        (['Eq(t0, fc[xi,yi] + fd[xi+1,yi+2])',
          'Eq(t1, fc[xi+1,yi+1] + fd[xi+2,yi+3])'],
         ['fc[xi+1,yi+1] + fd[xi+2,yi+3]']),
        # 2D with constant access
        (['Eq(t0, fc[x,y]*fc[x,0] + fd[x,y])',
          'Eq(t1, fc[x+1,y+1]*fc[x+1,0] + fd[x+1,y+1])'],
         ['fc[x+1,y+1]*fc[x+1,0] + fd[x+1,y+1]']),
        # 2D with multiple, non-zero, constant accesses
        (['Eq(t0, fc[x,y]*fc[x,0] + fd[x,y]*fc[x,1])',
          'Eq(t1, fc[x+1,y+1]*fc[x+1,0] + fd[x+1,y+1]*fc[x+1,1])'],
         ['fc[x+1,0]*fc[x+1,y+1] + fc[x+1,1]*fd[x+1,y+1]']),
        # 2D with different shapes
        (['Eq(t0, fc[x,y]*fa[x] + fd[x,y])',
          'Eq(t1, fc[x+1,y+1]*fa[x+1] + fd[x+1,y+1])'],
         ['fc[x+1,y+1]*fa[x+1] + fd[x+1,y+1]']),
        # complex (two 2D aliases with stride inducing relaxation)
        (['Eq(t0, fc[x,y] + fd[x+1,y+2])',
          'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])',
          'Eq(t2, fc[x+1,y+1]*3. + fd[x+2,y+2])',
          'Eq(t3, fc[x+2,y+2]*3. + fd[x+3,y+3])'],
         ['fc[x+1,y+1] + fd[x+2,y+3]', '3.*fc[x+2,y+2] + fd[x+3,y+3]']),
    ])
    def test_collection(self, exprs, expected):
        """
        Unit test for the detection and collection of aliases out of a series
        of input expressions.
        """
        grid = Grid(shape=(4, 4))
        x, y = grid.dimensions  # noqa
        xi, yi = grid.interior.dimensions  # noqa

        t0 = Scalar(name='t0')  # noqa
        t1 = Scalar(name='t1')  # noqa
        t2 = Scalar(name='t2')  # noqa
        t3 = Scalar(name='t3')  # noqa
        fa = Function(name='fa', grid=grid, shape=(4,), dimensions=(x,), space_order=4)  # noqa
        fb = Function(name='fb', grid=grid, shape=(4,), dimensions=(x,), space_order=4)  # noqa
        fc = Function(name='fc', grid=grid, space_order=4)  # noqa
        fd = Function(name='fd', grid=grid, space_order=4)  # noqa

        # List/dict comprehension would need explicit locals/globals mappings to eval
        for i, e in enumerate(list(exprs)):
            exprs[i] = DummyEq(indexify(eval(e).evaluate))
        for i, e in enumerate(list(expected)):
            expected[i] = eval(e)

        extracted = {i.rhs: i.lhs for i in exprs}
        ispace = exprs[0].ispace

        aliases = collect(extracted, ispace, False)

        assert len(aliases) == len(expected)
        assert all(i.pivot in expected for i in aliases)

    def get_params(self, op, *names):
        ret = []
        for i in names:
            for p in op.parameters:
                if i == p.name:
                    ret.append(p)
        return tuple(ret)

    def check_array(self, array, exp_halo, exp_shape, rotate=False):
        assert len(array.dimensions) == len(exp_halo)

        shape = []
        for i in array.symbolic_shape:
            if i.is_Number or i.is_Symbol:
                shape.append(i)
            else:
                assert i.is_Add
                shape.append(Add(*i.args))

        if rotate:
            exp_shape = (sum(exp_halo[0]) + 1,) + tuple(exp_shape[1:])
            exp_halo = ((0, 0),) + tuple(exp_halo[1:])

        assert tuple(array.halo) == exp_halo
        assert tuple(shape) == tuple(exp_shape)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_full_shape(self, rotate):
        """
        Check the shape of the Array used to store an aliasing expression.
        The shape is impacted by loop blocking, which reduces the required
        write-to space.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name="u1", grid=grid, space_order=3)

        u.data_with_halo[:] = 0.5
        u1.data_with_halo[:] = 0.5

        # Leads to 3D aliases
        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z+1])*3. +
                               _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.) + 1.)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        self.check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+2), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_contracted_shape(self, rotate):
        """
        Conceptually like `test_full_shape`, but the Operator used in this
        test leads to contracted Arrays (2D instead of 3D).
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        u.data_with_halo[:] = 0.5
        u1.data_with_halo[:] = 0.5

        # Leads to 2D aliases
        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x, y+1, z+1])*3. +
                               _R(u[t, x, y+2, z+2] + u[t, x, y+3, z+3])*3.) + 1)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        ys, zs = self.get_params(op1, 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        self.check_array(arrays[0], ((1, 1), (1, 1)), (ys+2, zs+2), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_uncontracted_shape(self, rotate):
        """
        Like `test_contracted_shape`, but the potential contraction is
        now along the innermost Dimension, which causes falling back to
        3D Arrays.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        u.data_with_halo[:] = 0.5
        u1.data_with_halo[:] = 0.5

        # Leads to 3D aliases
        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z])*3. +
                               _R(u[t, x+2, y+2, z] + u[t, x+3, y+3, z])*3.) + 1)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        self.check_array(arrays[0], ((1, 1), (1, 1), (0, 0)), (xs+2, ys+2, zs), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    def test_uncontracted_shape_invariants(self):
        """
        Like `test_uncontracted_shape`, but now with some (outer-)Dimension-invariant
        aliasing expressions.
        """
        grid = Grid(shape=(6, 6, 6))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        f.data_with_halo[:] =\
            np.linspace(-1, 1, f.data_with_halo.size).reshape(*f.shape_with_halo)
        u.data_with_halo[:] = 0.5
        u1.data_with_halo[:] = 0.5

        def func(f):
            return sqrt(f**2 + 1.)

        # Leads to 3D aliases despite the potential contraction along x and y
        eqn = Eq(u.forward, u*func(f) + u*func(f[x, y, z-1]))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True}))

        # Check code generation
        xs, ys, zs = self.get_params(op1, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 1
        self.check_array(arrays[0], ((0, 0), (0, 0), (1, 0)), (xs, ys, zs+1))

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.allclose(u.data, u1.data, rtol=10e-7)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_full_shape_w_subdims(self, rotate):
        """
        Like `test_full_shape`, but SubDomains (and therefore SubDimensions) are used.
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
                               _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.) + 1,
                 subdomain=grid.interior)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'i0x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'i0x0_blk0_size', 'i0y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['i0x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['i0x0_blk0'])) == 1
        self.check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+2), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_mixed_shapes(self, rotate):
        """
        Test that if running with ``opt=(..., {'min-storage': True})``, then,
        when possible, aliasing expressions are assigned to (n-k)D Arrays (k>0)
        rather than nD Arrays.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        d = Dimension(name='d')

        c = Function(name='c', grid=grid, shape=(2, 3), dimensions=(d, z))
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u', grid=grid, space_order=3)

        c.data_with_halo[:] = 1.
        u.data_with_halo[:] = 1.5
        u1.data_with_halo[:] = 1.5

        # Leads to 2D and 3D aliases
        eqn = Eq(u.forward,
                 _R(_R(c[0, z]*u[t, x+1, y+1, z] + c[1, z+1]*u[t, x+1, y+1, z+1]) +
                    _R(c[0, z]*u[t, x+2, y+2, z] + c[1, z+1]*u[t, x+2, y+2, z+1]) +
                    _R(u[t, x, y+1, z+1] + u[t, x+1, y+1, z+1]*3.) +
                    _R(u[t, x, y+3, z+1] + u[t, x+1, y+3, z+1]*3.)))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced',
                                 {'openmp': True, 'min-storage': True,
                                  'cire-mingain': 0, 'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        self.check_array(arrays[0], ((1, 0), (1, 0), (0, 0)), (xs+1, ys+1, zs), rotate)
        self.check_array(arrays[1], ((1, 1), (0, 0)), (ys+2, zs), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    def test_min_storage_in_isolation(self):
        """
        Test that if running with ``opt=('cire-sops', {'min-storage': True})``,
        then, when possible, aliasing expressions are assigned to (n-k)D Arrays (k>0)
        rather than nD Arrays.
        """
        grid = Grid(shape=(8, 8, 8))
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name="u1", grid=grid, space_order=4)
        u2 = TimeFunction(name="u2", grid=grid, space_order=4)

        u.data_with_halo[:] = 1.42
        u1.data_with_halo[:] = 1.42
        u2.data_with_halo[:] = 1.42

        eqn = Eq(u.forward, u.dy.dy + u.dx.dx)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('cire-sops', 'simd', {'openmp': True,
                                                       'min-storage': True}))
        op2 = Operator(eqn, opt=('advanced-fsg', {'openmp': True}))

        # NOTE: `op1` uses the `simd` pass as well simply so that the
        # parallelization heuristics, seeing that the innermost Iteration was
        # vectorized, stick to parallelizing the outermost loop instead of
        # the two inner loops (which would normally be preferred due to collapse(2))

        # Check code generation
        # `min-storage` leads to one 2D and one 3D Arrays
        xs, ys, zs = self.get_params(op1, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(op1)) == 1
        self.check_array(arrays[0], ((2, 2), (0, 0), (0, 0)), (xs+4, ys, zs))
        self.check_array(arrays[1], ((2, 2), (0, 0)), (ys+4, zs))

        # Check that `advanced-fsg` + `min-storage` is incompatible
        try:
            Operator(eqn, opt=('advanced-fsg', {'openmp': True, 'min-storage': True}))
        except InvalidOperator:
            assert True
        except:
            assert False

        # Check that `cire-rotate=True` has no effect in this code has there's
        # no blocking
        op3 = Operator(eqn, opt=('cire-sops', 'simd', {'openmp': True,
                                                       'min-storage': True,
                                                       'cire-rotate': True}))
        assert str(op3) == str(op1)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        op2(time_M=1, u=u2)
        expected = norm(u)
        assert np.isclose(expected, norm(u1), rtol=1e-5)
        assert np.isclose(expected, norm(u2), rtol=1e-5)

    def test_min_storage_issue_1506(self):
        grid = Grid(shape=(10, 10))

        u1 = TimeFunction(name='u1', grid=grid, time_order=2, space_order=4, save=10)
        u2 = TimeFunction(name='u2', grid=grid, time_order=2, space_order=4, save=10)
        v1 = TimeFunction(name='v1', grid=grid, time_order=2, space_order=4, save=None)
        v2 = TimeFunction(name='v2', grid=grid, time_order=2, space_order=4, save=None)

        eqns = [Eq(u1.forward, (u1+u2).laplace),
                Eq(u2.forward, (u1-u2).laplace),
                Eq(v1.forward, (v1+v2).laplace + u1.dt2),
                Eq(v2.forward, (v1-v2).laplace + u2.dt2)]

        op0 = Operator(eqns, opt=('advanced', {'min-storage': False, 'cire-mingain': 1}))
        op1 = Operator(eqns, opt=('advanced', {'min-storage': True, 'cire-mingain': 1}))

        # Check code generation
        assert len([i for i in FindSymbols().visit(op0) if i.is_Array]) == 4
        # In particular, check that `min-storage` works, but has "no effect" in this
        # example, in the sense that it produces the same code as default
        assert str(op0) == str(op1)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_mixed_shapes_v2_w_subdims(self, rotate):
        """
        Analogous `test_mixed_shapes`, but with different sets of aliasing expressions.
        Also, uses SubDimensions.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        d = Dimension(name='d')

        c = Function(name='c', grid=grid, shape=(2, 3), dimensions=(d, z))
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        c.data_with_halo[:] = 1.
        u.data_with_halo[:] = 1.5
        u1.data_with_halo[:] = 1.5

        # Leads to 2D and 3D aliases
        eqn = Eq(u.forward,
                 _R(_R(c[0, z]*u[t, x+1, y-1, z] + c[1, z+1]*u[t, x+1, y-1, z+1]) +
                    _R(c[0, z]*u[t, x+2, y-2, z] + c[1, z+1]*u[t, x+2, y-2, z+1]) +
                    _R(u[t, x, y+1, z+1] + u[t, x+1, y+1, z+1])*3. +
                    _R(u[t, x, y+3, z+2] + u[t, x+1, y+3, z+2])*3.),
                 subdomain=grid.interior)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced',
                                 {'openmp': True, 'min-storage': True,
                                  'cire-mingain': 0, 'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'i0x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'i0x0_blk0_size', 'i0y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['i0x0_blk0']) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(pbs['i0x0_blk0'])) == 2
        self.check_array(arrays[0], ((1, 0), (1, 0), (0, 0)), (xs+1, ys+1, zs), rotate)
        self.check_array(arrays[1], ((1, 1), (1, 0)), (ys+2, zs+1), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_in_bounds_w_shift(self, rotate):
        """
        Make sure the iteration space and indexing of the aliasing expressions
        are shifted such that no out-of-bounds accesses are generated.
        """
        grid = Grid(shape=(5, 5, 5))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        d = Dimension(name='d')

        c = Function(name='c', grid=grid, shape=(2, 5), dimensions=(d, z))
        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name='u1', grid=grid, space_order=4)

        c.data_with_halo[:] = 1.
        u.data_with_halo[:] = 1.5
        u1.data_with_halo[:] = 1.5

        # Leads to 3D aliases
        eqn = Eq(u.forward,
                 _R(_R(c[0, z]*u[t, x+1, y, z] + c[1, z+1]*u[t, x+1, y, z+1]) +
                    _R(c[0, z]*u[t, x+2, y+2, z] + c[1, z+1]*u[t, x+2, y+2, z+1]) +
                    _R(u[t, x, y-4, z+1] + u[t, x+1, y-4, z+1])*3. +
                    _R(u[t, x-1, y-3, z+1] + u[t, x, y-3, z+1])*3.))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        self.check_array(arrays[0], ((1, 0), (1, 1), (0, 0)), (xs+1, ys+2, zs), rotate)
        self.check_array(arrays[1], ((1, 0), (1, 1), (0, 0)), (xs+1, ys+2, zs), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_constant_symbolic_distance(self, rotate):
        """
        Test the detection of aliasing expressions in the case of a
        constant symbolic distance, such as `a[t, x_m+2, y, z]` when the
        Dimensions are `(t, x, y, z)`; here, `x_m + 2` is a constant
        symbolic access.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        x_m = x.symbolic_min
        y_m = y.symbolic_min
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        u.data_with_halo[:] = 0.5
        u1.data_with_halo[:] = 0.5

        # Leads to 2D aliases
        eqn = Eq(u.forward,
                 _R(_R(u[t, x_m+2, y, z] + u[t, x_m+3, y+1, z+1])*3. +
                    _R(u[t, x_m+2, y+2, z+2] + u[t, x_m+3, y+3, z+3])*3. + 1 +
                    _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3. +  # N, not an alias
                    _R(u[t, x_m+1, y+2, z+2] + u[t, x_m+1, y+3, z+3])*3. +  # Y, redundant
                    _R(u[t, x+2, y_m+3, z+2] + u[t, x+3, y_m+3, z+3])*3. +
                    _R(u[t, x+1, y_m+3, z+1] + u[t, x+2, y_m+3, z+2])*3.))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 3
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3
        self.check_array(arrays[0], ((1, 0), (1, 0)), (xs+1, zs+1), rotate)
        self.check_array(arrays[1], ((1, 1), (1, 1)), (ys+2, zs+2), rotate)
        self.check_array(arrays[2], ((1, 1), (1, 1)), (ys+2, zs+2), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_outlier_with_long_diameter(self, rotate):
        """
        Test that if there is a potentially aliasing expression, say A, with
        excessively long diameter (that is, such that it cannot safely be
        computed in a loop with other aliasing expressions), then A is ignored
        and the other aliasing expressions are captured correctly.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)

        u.data_with_halo[:] = 1.5
        u1.data_with_halo[:] = 1.5

        # Leads to 3D aliases
        # Note: the outlier already touches the halo extremes, so it cannot
        # be computed in a loop with extra y-iterations, hence it must be ignored
        # while not compromising the detection of the two aliasing sub-expressions
        eqn = Eq(u.forward, _R(_R(u[t, x, y+1, z+1] + u[t, x+1, y+1, z+1])*3. +
                               _R(u[t, x, y-3, z+1] + u[t, x+1, y+3, z+1])*3. +  # outlier
                               _R(u[t, x, y+3, z+2] + u[t, x+1, y+3, z+2])*3.))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        ys, zs = self.get_params(op1, 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        self.check_array(arrays[0], ((1, 1), (1, 0)), (ys+2, zs+1), rotate)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    def test_take_largest_derivative(self):
        """
        Check that CIRE is able to automatically schedule the largest degree
        derivative in a case with many nested derivatives.
        """
        grid = Grid(shape=(4, 4, 4))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=2)

        eq = Eq(u.forward, f**2*sin(f)*u.dy.dy.dy.dy.dy)

        op = Operator(eq, opt=('cire-sops'))

        assert op._profiler._sections['section0'].sops == 84
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 1

    def test_nested_invariants_v1(self):
        """
        Check that nested aliases are optimized away.
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions  # noqa

        u = TimeFunction(name='u', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator(Eq(u.forward, u + sin(cos(g)) + sin(cos(g[x+1, y+1]))))

        # We expect one temporary Array: `r0 = sin(cos(g))`
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 1
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_nested_invariants_v2(self):
        """
        Check that nested aliases are optimized away.
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions  # noqa
        h_x, h_y = grid.spacing_symbols

        u = TimeFunction(name='u', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator(Eq(u.forward, u + (1 + cos(g))/h_x + (1 + cos(g[x+1, y+1]))/h_y))

        # We expect one temporary Array: `r0 = cos(g)`
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 1
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_nested_invariants_v3(self):
        """
        Check that nested aliases are optimized away.
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions  # noqa
        h_x, h_y = grid.spacing_symbols

        u = TimeFunction(name='u', grid=grid)
        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator(Eq(u.forward, (u*sin(f) + 1)*cos(g)*sin(g)))

        # We expect two temporary Arrays: `r0 = sin(f)` and `r1 = cos(g)*sin(g)`
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)
        # Also make sure the inner `sin` has been correctly replaced
        exprs = FindNodes(Expression).visit(op)
        assert len(exprs[-1].expr.find(sin)) == 0

    def test_nested_invariant_v4(self):
        """
        Check that nested aliases are optimized away.
        """
        grid = Grid((10, 10))

        a = Function(name="a", grid=grid, space_order=4)
        b = Function(name="b", grid=grid, space_order=4)

        e = TimeFunction(name="e", grid=grid, space_order=4)
        f = TimeFunction(name="f", grid=grid, space_order=4)

        subexpr0 = sqrt(1. + 1./a)
        subexpr1 = 1/(8.*subexpr0 - 8./b)
        eqns = [Eq(e.forward, e + 1),
                Eq(f.forward, f*subexpr0 - f*subexpr1 + e.forward.dx)]

        op = Operator(eqns)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 3
        arrays = [i for i in FindSymbols().visit(trees[0].root) if i.is_Array]
        assert len(arrays) == 1
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_nested_invariant_v5(self):
        """
        Check that nested aliases are optimized away.
        """
        grid = Grid((10, 10))
        x, y = grid.dimensions
        hx = x.spacing
        dt = grid.stepping_dim.spacing

        xright = SubDimension.right(name='xright', parent=x, thickness=4)

        a = Function(name="a", grid=grid)
        b = Function(name="b", grid=grid)
        e = TimeFunction(name="e", grid=grid)

        expr = 1./(5.*dt*sqrt(a)*b/hx + 2.*dt**2*b**2*a/hx**2 + 3.)
        eq = Eq(e.forward, 2.*expr*sqrt(a) + 3.*expr + e*sqrt(a)).subs({x: xright})

        op = Operator(eq, openmp=False)

        # Check generated code
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 1
        exprs = FindNodes(Expression).visit(op)
        assert len(exprs) == 2
        assert exprs[0].write is arrays[0]
        assert exprs[0].expr.rhs.is_Pow

    @switchconfig(profiling='advanced')
    def test_twin_sops(self):
        """
        Check that identical sum-of-product aliases are caught via CSE thus
        reducing the operation count (but not the working set size).
        """
        grid = Grid(shape=(10, 10, 10), dtype=np.float64)
        x, y, z = grid.dimensions

        space_order = 2
        u = TimeFunction(name='u', grid=grid, space_order=space_order)
        v = TimeFunction(name='v', grid=grid, space_order=space_order)
        u1 = TimeFunction(name='u', grid=grid, space_order=space_order)
        v1 = TimeFunction(name='v', grid=grid, space_order=space_order)
        f = Function(name='f', grid=grid, space_order=space_order)
        e = Function(name='e', grid=grid, space_order=space_order)
        p0 = Function(name='p0', grid=grid, space_order=space_order)
        p1 = Function(name='p1', grid=grid, space_order=space_order)

        f.data[:] = 1.2
        e.data[:] = 0.3
        p0.data[:] = 0.4
        p1.data[:] = 0.7

        def d0(field):
            return (sin(p0) * cos(p1) * field.dx(x0=x+x.spacing/2) +
                    sin(p0) * sin(p1) * field.dy(x0=y+y.spacing/2) +
                    cos(p0) * field.dz(x0=z+z.spacing/2))

        def d1(field):
            return ((sin(p0) * cos(p1) * field).dx(x0=x-x.spacing/2) +
                    (sin(p0) * sin(p1) * field).dy(x0=y-y.spacing/2) +
                    (cos(p0) * field).dz(x0=z-z.spacing/2))

        eqns = [Eq(u.forward, d1((1 - f * e**2) + f * e * sqrt(1 - e**2) * d0(v))),
                Eq(v.forward, d1((1 - f + f * e**2) * d0(v) + f * e * sqrt(1 - e**2)))]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt=('advanced', {'cire-schedule': 2}))

        # Check code generation
        # We expect two temporary Arrays which have in common a sub-expression
        # stemming from `d0(v, p0, p1)`
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 6
        vexpandeds = FindNodes(VExpanded).visit(pbs['x0_blk0'])
        assert len(vexpandeds) == (2 if configuration['language'] == 'openmp' else 0)
        assert all(i._mem_heap and not i._mem_external for i in arrays)
        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 2
        exprs = FindNodes(Expression).visit(trees[0][2])
        assert exprs[-1].write is arrays[-1]
        assert arrays[-2] not in exprs[-1].reads

        # Check numerical output
        op0(time_M=2)
        summary1 = op1(time_M=2, u=u1, v=v1)
        assert np.isclose(norm(u), norm(u1), rtol=10e-16)
        assert np.isclose(norm(v), norm(v1), rtol=10e-16)

        # Also check against expected operation count to make sure
        # all redundancies have been detected correctly
        assert sum(i.ops for i in summary1.values()) == 69

    @pytest.mark.parametrize('rotate', [False, True])
    def test_from_different_nests(self, rotate):
        """
        Check that aliases arising from two sets of equations A and B,
        characterized by a flow dependence, are scheduled within A's and B's
        loop nests respectively.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim
        i = Dimension(name='i')

        g = Function(name='g', shape=(3,), dimensions=(i,))
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name='u1', grid=grid, space_order=3)
        v = TimeFunction(name='v', grid=grid, space_order=3)
        v1 = TimeFunction(name='v1', grid=grid, space_order=3)

        uf = u.forward
        vf = v.forward

        g.data[:] = 2.

        # Leads to 3D aliases
        eqns = [Eq(uf, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z+1])*3. +
                          _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3. + 1.)),
                Inc(u[t+1, i, i, i], g + 1),
                Eq(vf, _R(_R(v[t, x, y, z] + v[t, x+1, y+1, z+1])*3. +
                          _R(v[t, x+2, y+2, z+2] + v[t, x+3, y+3, z+3])*3. + 1.) + uf)]
        op0 = Operator(eqns, opt=('noop', {'openmp': True}))
        op1 = Operator(eqns, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                               'cire-rotate': rotate}))

        # Check code generation
        bns, _ = assert_blocking(op1, {'x0_blk0', 'x1_blk0'})
        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].body[0].write.is_Array
        assert trees[1][-1].nodes[0].body[0].write is u
        trees = retrieve_iteration_tree(bns['x1_blk0'])
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].body[0].write.is_Array
        assert trees[1][-1].nodes[0].body[0].write is v

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1, v=v1)
        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    @switchconfig(autopadding=True, platform='knl7210')  # Platform is to fix pad value
    def test_minimize_remainders_due_to_autopadding(self, rotate):
        """
        Check that the bounds of the Iteration computing an aliasing expression are
        relaxed (i.e., slightly larger) so that backend-compiler-generated remainder
        loops are avoided.
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

        op0 = Operator(eqn, opt=('noop', {'openmp': False}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': False, 'cire-mingain': 0,
                                              'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 0
        assert arrays[0].padding == ((0, 0), (0, 0), (0, 30))
        self.check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+32), rotate)
        # Check loop bounds
        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 2
        expected_rounded = trees[0].inner
        assert expected_rounded.symbolic_max ==\
            z.symbolic_max + (z.symbolic_max - z.symbolic_min + 3) % 16 + 1

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    def test_catch_best_invariant_v1(self):
        """
        Make sure the best time-invariant sub-expressions are extracted.
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions  # noqa

        g = Function(name='g', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        u1 = TimeFunction(name='u1', grid=grid)

        g.data[:] = 2.
        u.data[:] = 1.
        u1.data[:] = 1.

        expr = (cos(g)*cos(g) +
                sin(g)*sin(g) +
                sin(g)*cos(g) +
                sin(g[x + 1, y + 1])*cos(g[x + 1, y + 1]))*u

        op0 = Operator(Eq(u.forward, expr), opt='noop')
        op1 = Operator(Eq(u.forward, expr))

        # Check code generation
        # We expect two temporary Arrays, one for each trascendental function
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.allclose(u.data, u1.data, rtol=10e-7)

    def test_catch_best_invariant_v2(self):
        """
        Make sure the best time-invariant sub-expressions are extracted.
        """
        grid = Grid((10, 10))

        a = Function(name="a", grid=grid, space_order=4)
        b = Function(name="b", grid=grid, space_order=4)
        c = Function(name="c", grid=grid, space_order=4)
        d = Function(name="d", grid=grid, space_order=4)

        e = TimeFunction(name="e", grid=grid, space_order=4)

        deriv = (sqrt((a - 2*b)/c) * e.dx).dy + (sqrt((d - 2*c)/a) * e.dy).dx

        op = Operator(Eq(e.forward, deriv + e))

        # We expect four temporary Arrays, two of which for the `sqrt` subexpr
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 4

        exprs = FindNodes(Expression).visit(op)
        sqrt_exprs = exprs[2:4]
        assert all(e.write in arrays for e in sqrt_exprs)
        assert all(e.expr.rhs.is_Pow for e in sqrt_exprs)
        assert all(e.write._mem_heap and not e.write._mem_external for e in sqrt_exprs)

        tmp_exprs = exprs[4:6]
        assert all(e.write in arrays for e in tmp_exprs)
        assert all(e.write._mem_heap and not e.write._mem_external for e in tmp_exprs)

    def test_compound_invariants(self):
        """
        Check that compound time-invariant aliases are optimized away.
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        u1 = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        v1 = TimeFunction(name='v', grid=grid)

        f.data[:] = 1.4
        g.data[:] = 2.1
        u.data[:] = 1.3
        u1.data[:] = 1.3
        v.data[:] = 1.7
        v1.data[:] = 1.7

        eqn = Eq(u.forward, (cos(f)*sin(g)*u +
                             cos(g)*sin(f)*v +
                             cos(f[x+1, y+1])*sin(g[x+1, y+1])*u[t, x+1, y+1]))

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn)

        # Check code generation
        # We expect two temporary Arrays, one for cos(f)*sin(g) and one for cos(g)*sin(f)
        # Old versions of devito would have used four Arrays, respectively for cos(f),
        # cos(g), sin(f), sin(g)
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1, v=v1)
        assert np.allclose(u.data, u1.data, rtol=10e-7)

    def test_space_invariant(self):
        """
        Unlike most cases, here a sub-expression is invariant w.r.t. space, but
        not to time and/or frequency.
        """
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        time = grid.time_dim

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=8)

        f = DefaultDimension(name="f", default_value=10)
        freq = Function(name="freq", dimensions=(f,), shape=(10,))
        uf = Function(name="uf", dimensions=(f, x, y), shape=(10, 401, 401))

        pde = Eq(u.forward, 2*u - u.backward + u.laplace)
        df = Inc(uf, 2*u*cos(time*freq))

        op = Operator([pde, df])

        # Check code generation
        assert_structure(op, ['t,x,y', 't,f', 't,f0_blk0,f,x,y'],
                         't,x,y,f,f0_blk0,f,x,y')

    def test_space_invariant_v2(self):
        """
        Similar to test_space_invariant, but now the invariance is only w.r.t.
        one of the inner space dimensions.
        """
        grid1 = Grid(shape=(10, 10, 10))
        x, y, z = grid1.dimensions
        grid2 = Grid(shape=(10, 10), dimensions=(x, y))

        u1 = TimeFunction(name="u", grid=grid1)
        u2 = TimeFunction(name="u", grid=grid2)

        for u in [u1, u2]:
            eq = Eq(u.forward, u*sin(y + y.symbolic_max))

            op = Operator(eq)

            # Check code generation
            ys = self.get_params(op, 'y_size')[0]
            arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
            assert len(arrays) == 1
            self.check_array(arrays[0], ((0, 0),), (ys,))
            trees = retrieve_iteration_tree(op)
            assert len(trees) == 2
            assert trees[0].root.dim is y

    def test_space_invariant_v3(self):
        """
        Similar to test_space_invariant, but now with many invariants along
        different subsets of space dimensions.
        """
        grid = Grid(shape=(10, 10, 10))
        x, y, z = grid.dimensions

        f = Function(name='f', grid=grid)

        eq = Eq(f, f + cos(x*y*z) + cos(x*y)*cos(y) + sin(x)*cos(x*z))

        op = Operator(eq)

        xs, ys, zs = self.get_params(op, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 3
        self.check_array(arrays[0], ((0, 0),), (ys,))
        self.check_array(arrays[1], ((0, 0), (0, 0)), (xs, zs))
        self.check_array(arrays[2], ((0, 0), (0, 0)), (xs, ys))

    def test_space_invariant_v4(self):
        """
        Similar to test_space_invariant, stems from viscoacoustic -- a portion
        of a space derivative that would be redundantly computed in two separated
        loop nests is recognised to be a time invariant and factored into a common
        temporary.
        """
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)
        v = TimeFunction(name='v', grid=grid)

        eqns = [Eq(u.forward, (u*cos(f)).dx + v),
                Eq(v.forward, (v*cos(f)).dy + u.forward.dx)]

        op = Operator(eqns)

        xs, ys, zs = self.get_params(op, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 1
        self.check_array(arrays[0], ((1, 0), (1, 0), (0, 0)), (xs+1, ys+1, zs))
        assert op._profiler._sections['section1'].sops == 15

    def test_catch_duplicate_from_different_clusters(self):
        """
        Check that the compiler is able to detect redundant aliases when these
        stem from different Clusters.
        """
        grid = Grid((10, 10))

        a = Function(name="a", grid=grid, space_order=4)
        b = Function(name="b", grid=grid, space_order=4)
        c = Function(name="c", grid=grid, space_order=4)
        d = Function(name="d", grid=grid, space_order=4)

        s = SparseTimeFunction(name="s", grid=grid, npoint=1, nt=2)
        e = TimeFunction(name="e", grid=grid, space_order=4)
        f = TimeFunction(name="f", grid=grid, space_order=4)

        deriv = (sqrt((a - 2*b)/c) * e.dx).dy + (sqrt((d - 2*c)/a) * e.dy).dx
        deriv2 = (sqrt((c - 2*b)/c) * f.dy).dx + (sqrt((d - 2*c)/a) * f.dx).dy

        eqns = ([Eq(e.forward, deriv + e)] +
                s.inject(e.forward, expr=s) +
                [Eq(f.forward, deriv2 + f + e.forward.dx)])

        op = Operator(eqns, opt=('advanced', {'cire-mingain': 100}))

        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 3
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_discarded_compound(self):
        """
        Test that compound aliases may be ignored if part of a bigger alias.
        """
        grid = Grid((10, 10))
        dt = grid.time_dim.spacing

        a = Function(name="a", grid=grid, space_order=4)
        e = TimeFunction(name="e", grid=grid, space_order=4)

        eqn = Eq(e.forward, e/(1./cos(a) + 1/(dt**2*a**2)) + dt**-2 + a**-2)

        op = Operator(eqn)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        arrays = [i for i in FindSymbols().visit(trees[0].root) if i.is_Array]
        assert len(arrays) == 1
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_lazy_solve_produces_larger_temps(self):
        """
        Test that using `solve` doesn't affect CIRE.
        """
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name="u", grid=grid, space_order=4, time_order=2)

        pde = u.dt2 - (u.dx.dx + u.dy.dy) + u.dx.dy
        eq = Eq(u.forward, solve(pde, u.forward))

        op = Operator(eq)
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 2
        assert op._profiler._sections['section0'].sops == 39

    def test_hoisting_iso_ot4_akin(self):
        """
        Test hoisting of time invariant sub-expressions in iso-acoustic-like kernels.
        """
        grid = Grid(shape=(3, 3, 3))
        s = grid.time_dim.spacing

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)
        m = Function(name='m', grid=grid, space_order=4)

        # The Eq implements an OT2 iso-acoustic stencil
        pde = m * u.dt2 - u.laplace
        eq = Eq(u.forward, solve(pde, u.forward))

        op = Operator(eq, opt=('advanced', {'openmp': False}))
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 0
        assert op._profiler._sections['section0'].sops == 34

        # The Eq implements an OT4 iso-acoustic stencil
        pde = m * u.dt2 - u.laplace - s**2/12 * u.biharmonic(1/m)
        eq = Eq(u.forward, solve(pde, u.forward))

        op0 = Operator(eq, opt=('advanced', {'openmp': False}))
        assert len([i for i in FindSymbols().visit(op0) if i.is_Array]) == 2
        assert op0._profiler._sections['section1'].sops == 62

        op1 = Operator(eq, opt=('advanced', {'openmp': False}),
                       subs={i: 0.5 for i in grid.spacing_symbols})
        assert len([i for i in FindSymbols().visit(op1) if i.is_Array]) == 2
        assert op1._profiler._sections['section1'].sops == 44

    def test_hoisting_scalar_divs(self):
        """
        Test that scalar divisions are hoisted out of the inner loops.
        """
        grid = Grid(shape=(3, 3))

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)
        m = Function(name='m', grid=grid, space_order=4)

        pde = m * u.dt2 - u.laplace
        eq = Eq(u.forward, solve(pde, u.forward))

        # Check that different backends behave the same
        op0 = Operator(eq, opt=('advanced', {'openmp': False}))
        op1 = Operator(eq, platform='nvidiaX', language='openacc')

        for op in [op0, op1]:
            assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 0
            assert op._profiler._sections['section0'].sops == 26
            exprs = FindNodes(Expression).visit(op)
            assert len(exprs) == 5
            assert all(e.is_scalar for e in exprs[:-1])
            assert op.body.body[-1].body[0].is_ExpressionBundle
            assert op.body.body[-1].body[-1].is_Iteration

    @pytest.mark.parametrize('rotate', [False, True])
    def test_drop_redundants_after_fusion(self, rotate):
        """
        Test for detection of redundant aliases that get exposed after
        Cluster fusion.
        """
        grid = Grid(shape=(10, 10))

        t = cos(Function(name="t", grid=grid))
        p = sin(Function(name="p", grid=grid))

        a = TimeFunction(name="a", grid=grid)
        b = TimeFunction(name="b", grid=grid)
        c = TimeFunction(name="c", grid=grid)
        d = TimeFunction(name="d", grid=grid)
        e = TimeFunction(name="e", grid=grid)
        f = TimeFunction(name="f", grid=grid)

        s1 = SparseTimeFunction(name="s1", grid=grid, npoint=1, nt=2)

        eqns = [Eq(a.forward, t*a.dx + p*b.dy),
                Eq(b.forward, p*b.dx + p*t*a.dy)]

        eqns += s1.inject(field=a.forward, expr=s1)
        eqns += s1.inject(field=b.forward, expr=s1)

        eqns += [Eq(c.forward, t*p*a.forward.dx + b.forward.dy),
                 Eq(d.forward, t*d.dx + e.dy + p*a.dt),
                 Eq(e.forward, p*d.dx + e.dy + t*b.dt)]

        eqns += [Eq(f.forward, t*p*e.forward.dx + p*d.forward.dy)]

        op = Operator(eqns, opt=('advanced', {'cire-rotate': rotate}))

        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)

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
        op1 = Operator(eqn, opt=('advanced-fsg', {'openmp': True, 'cire-mingain': 0}))

        # Check code generation
        bns, _ = assert_blocking(op1, {'x0_blk0', 'x1_blk0'})
        xs, ys, zs = self.get_params(op1, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        self.check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+2))

        # Check that `cire-rotate=True` has no effect in this code has there's
        # no cross-loop blocking
        op2 = Operator(eqn, opt=('advanced-fsg', {'openmp': True, 'cire-rotate': True,
                                                  'cire-mingain': 0}))
        assert str(op2) == str(op1)

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1)
        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('rotate', [False, True])
    @switchconfig(profiling='advanced')
    def test_extraction_from_lifted_ispace(self, rotate):
        """
        Test that the aliases are scheduled correctly when extracted from
        Clusters whose iteration space is lifted (ie, stamp != 0).
        """
        so = 8
        grid = Grid(shape=(6, 6, 6))

        f = Function(name='f', grid=grid, space_order=so, is_param=True)
        v = TimeFunction(name="v", grid=grid, space_order=so)
        v1 = TimeFunction(name="v1", grid=grid, space_order=so)
        p = TimeFunction(name="p", grid=grid, space_order=so, staggered=NODE)
        p1 = TimeFunction(name="p1", grid=grid, space_order=so, staggered=NODE)

        v.data_with_halo[:] = 1.
        v1.data_with_halo[:] = 1.
        p.data_with_halo[:] = 0.5
        p1.data_with_halo[:] = 0.5
        f.data_with_halo[:] = 0.2

        eqns = [Eq(v.forward, v - f*p),
                Eq(p.forward, p - v.forward.dx + div(f*grad(p)))]

        # Operator
        op0 = Operator(eqns, opt=('noop', {'openmp': True}))
        op1 = Operator(eqns, opt=('advanced', {'openmp': True, 'cire-mingain': 1,
                                               'cire-rotate': rotate}))

        # Check numerical output
        op0(time_M=1)
        summary = op1(time_M=1, v=v1, p=p1)
        assert np.isclose(norm(v), norm(v1), rtol=1e-5)
        assert np.isclose(norm(p), norm(p1), atol=1e-5)

        # Also check against expected operation count to make sure
        # all redundancies have been detected correctly
        assert summary[('section0', None)].ops == 93

    @pytest.mark.parametrize('so_ops', [(4, 108)])
    @switchconfig(profiling='advanced')
    def test_tti_J_akin_bb0(self, so_ops):
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        space_order, exp_ops = so_ops

        g = Function(name='g', grid=grid, space_order=space_order)
        phi = Function(name='phi', grid=grid, space_order=space_order)
        p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
        m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)

        def g1(field):
            return (field.dx(x0=x+x.spacing/2) +
                    field.dy(x0=y+y.spacing/2) -
                    field.dz(x0=z+z.spacing/2))

        def g1_tilde(field, phi):
            return ((cos(phi) * field).dx(x0=x-x.spacing/2) +
                    (sin(phi) * field).dy(x0=y-y.spacing/2) -
                    field.dz(x0=z-z.spacing/2))

        update_p = g + \
            (g1_tilde(g1(p0), phi) +
             g1_tilde(g1(p0) + g1(m0), phi))

        eqn = Eq(p0.forward, update_p)

        op = Operator(eqn, subs=grid.spacing_map, openmp=True)

        # Check code generation
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        assert op._profiler._sections['section1'].sops == exp_ops
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 5
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3
        xs, ys, zs = self.get_params(op, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        # The three kind of derivatives taken -- in x, y, z -- all are over
        # different expressions, so this leads to three temporaries of dimensionality,
        # in particular 3D for the x-derivative, 2D for the y-derivative, and 1D
        # for the z-derivative
        self.check_array(arrays[2], ((2, 1), (0, 0), (0, 0)), (xs+3, ys, zs))
        self.check_array(arrays[3], ((2, 1), (0, 0)), (ys+3, zs))
        self.check_array(arrays[4], ((2, 1),), (zs+3,))

    @pytest.mark.parametrize('so_ops', [(4, 51), (8, 95)])
    @switchconfig(profiling='advanced')
    def test_tti_J_akin_bb1(self, so_ops):
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        space_order, exp_ops = so_ops

        vel = Function(name='vel', grid=grid, space_order=space_order)
        a = Function(name='a', grid=grid, space_order=space_order)
        phi = Function(name='phi', grid=grid, space_order=space_order)
        p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
        m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)

        def g1(field):
            return field.dx + field.dy - field.dz

        def g1_tilde(field, phi):
            return (cos(phi) * field).dx + (sin(phi) * field).dy - field.dz

        def g3_tilde(field, phi):
            return (cos(phi) * field).dx + (sin(phi) * field).dy + field.dz

        update_p = vel**2 * \
            (g1_tilde(g1(p0), phi) +
             g3_tilde(g1(p0) + sqrt(1 - vel**2) * g1(m0), phi)) + a

        eqn = Eq(p0.forward, update_p)

        op = Operator(eqn, subs=grid.spacing_map, openmp=True)

        # Check code generation
        assert op._profiler._sections['section1'].sops == exp_ops
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 6
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3

    @pytest.mark.parametrize('so_ops', [(4, 48)])
    @switchconfig(profiling='advanced')
    def test_tti_J_akin_bb2(self, so_ops):
        grid = Grid(shape=(16, 16, 16))
        x, y, z = grid.dimensions

        space_order, exp_ops = so_ops

        f = Function(name='f', grid=grid, space_order=space_order)
        theta = Function(name='theta', grid=grid, space_order=space_order)
        phi = Function(name='phi', grid=grid, space_order=space_order)
        p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)

        def g1(field, phi, theta):
            return (cos(theta) * cos(phi) * field.dx(x0=x+x.spacing/2) +
                    cos(theta) * sin(phi) * field.dy(x0=y+y.spacing/2) -
                    sin(theta) * field.dz(x0=z+z.spacing/2))

        def g2(field, phi, theta):
            return - (sin(phi) * field.dx(x0=x+x.spacing/2) -
                      cos(phi) * field.dy(x0=y+y.spacing/2))

        def g1_tilde(field, phi, theta):
            return ((cos(theta) * cos(phi) * field).dx(x0=x-x.spacing/2) +
                    (cos(theta) * sin(phi) * field).dy(x0=y-y.spacing/2) -
                    (sin(theta) * field).dz(x0=z-z.spacing/2))

        def g2_tilde(field, phi, theta):
            return - ((sin(phi) * field).dx(x0=x-x.spacing/2) -
                      (cos(phi) * field).dy(x0=y-y.spacing/2))

        update_p = exp_ops + f * (g1_tilde(g1(p0, phi, theta), phi, theta) +
                                  g2_tilde(g2(p0, phi, theta), phi, theta))

        eqn = Eq(p0.forward, update_p)

        op = Operator(eqn, subs=grid.spacing_map, openmp=True)

        # Check code generation
        assert op._profiler._sections['section1'].sops == exp_ops
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 7
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3

    @pytest.mark.parametrize('so_ops', [(4, 144), (8, 208)])
    @switchconfig(profiling='advanced')
    def test_tti_J_akin_complete(self, so_ops):
        grid = Grid(shape=(16, 16, 16))
        t = grid.stepping_dim
        x, y, z = grid.dimensions

        space_order, exp_ops = so_ops

        a = Function(name='a', grid=grid, space_order=space_order)
        f = Function(name='f', grid=grid, space_order=space_order)
        theta = Function(name='theta', grid=grid, space_order=space_order)
        phi = Function(name='phi', grid=grid, space_order=space_order)
        p0 = TimeFunction(name='p0', grid=grid, time_order=2, space_order=space_order)
        m0 = TimeFunction(name='m0', grid=grid, time_order=2, space_order=space_order)

        def g1(field, phi, theta):
            return (cos(theta) * cos(phi) * field.dx(x0=x+x.spacing/2) +
                    cos(theta) * sin(phi) * field.dy(x0=y+y.spacing/2) -
                    sin(theta) * field.dz(x0=z+z.spacing/2))

        def g2(field, phi, theta):
            return - (sin(phi) * field.dx(x0=x+x.spacing/2) -
                      cos(phi) * field.dy(x0=y+y.spacing/2))

        def g3(field, phi, theta):
            return (sin(theta) * cos(phi) * field.dx(x0=x+x.spacing/2) +
                    sin(theta) * sin(phi) * field.dy(x0=y+y.spacing/2) +
                    cos(theta) * field.dz(x0=z+z.spacing/2))

        def g1_tilde(field, phi, theta):
            return ((cos(theta) * cos(phi) * field).dx(x0=x-x.spacing/2) +
                    (cos(theta) * sin(phi) * field).dy(x0=y-y.spacing/2) -
                    (sin(theta) * field).dz(x0=z-z.spacing/2))

        def g2_tilde(field, phi, theta):
            return - ((sin(phi) * field).dx(x0=x-x.spacing/2) -
                      (cos(phi) * field).dy(x0=y-y.spacing/2))

        def g3_tilde(field, phi, theta):
            return ((sin(theta) * cos(phi) * field).dx(x0=x-x.spacing/2) +
                    (sin(theta) * sin(phi) * field).dy(x0=y-y.spacing/2) +
                    (cos(theta) * field).dz(x0=z-z.spacing/2))

        update_p = t.spacing**2 * a**2 / f * \
            (g1_tilde(f * g1(p0, phi, theta), phi, theta) +
             g2_tilde(f * g2(p0, phi, theta), phi, theta) +
             g3_tilde(f * g3(p0, phi, theta) + f * g3(m0, phi, theta), phi, theta)) + \
            (2 - t.spacing * a)

        update_m = t.spacing**2 * a**2 / f * \
            (g1_tilde(f * g1(m0, phi, theta), phi, theta) +
             g2_tilde(f * g2(m0, phi, theta), phi, theta) +
             g3_tilde(f * g3(m0, phi, theta) + f * g3(p0, phi, theta), phi, theta)) + \
            (2 - t.spacing * a)

        eqns = [Eq(p0.forward, update_p),
                Eq(m0.forward, update_m)]

        op = Operator(eqns, subs=grid.spacing_map, openmp=True)

        # Check code generation
        assert op._profiler._sections['section1'].sops == exp_ops
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 10
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 6

    @pytest.mark.parametrize('so_ops', [(4, 33), (8, 69)])
    @pytest.mark.parametrize('rotate', [False, True])
    @switchconfig(profiling='advanced')
    def test_tti_adjoint_akin(self, so_ops, rotate):
        """
        Extrapolated from TTI adjoint.
        """
        so, exp_ops = so_ops
        to = 2
        soh = so // 2
        T = transpose

        grid = Grid(shape=(10, 10, 10), dtype=np.float64)
        x, y, z = grid.dimensions

        p = TimeFunction(name='p', grid=grid, space_order=so, time_order=to)
        r = TimeFunction(name='r', grid=grid, space_order=so, time_order=to)
        r1 = TimeFunction(name='r1', grid=grid, space_order=so, time_order=to)
        delta = Function(name='delta', grid=grid, space_order=so)
        theta = Function(name='theta', grid=grid, space_order=so)
        phi = Function(name='phi', grid=grid, space_order=so)

        p.data_with_halo[:] = 1.
        r.data_with_halo[:] = 0.5
        r1.data_with_halo[:] = 0.5
        delta.data_with_halo[:] = 0.2
        theta.data_with_halo[:] = 0.8
        phi.data_with_halo[:] = 0.2

        costheta = cos(theta)
        sintheta = sin(theta)
        cosphi = cos(phi)
        sinphi = sin(phi)

        delta = sqrt(delta)

        field = delta*p + r
        Gz = -(sintheta * cosphi*first_derivative(field, dim=x, fd_order=soh) +
               sintheta * sinphi*first_derivative(field, dim=y, fd_order=soh) +
               costheta * first_derivative(field, dim=z, fd_order=soh))
        Gzz = (first_derivative(Gz * sintheta * cosphi, dim=x, fd_order=soh, matvec=T) +
               first_derivative(Gz * sintheta * sinphi, dim=y, fd_order=soh, matvec=T) +
               first_derivative(Gz * costheta, dim=z, fd_order=soh, matvec=T))

        # Equation
        eqn = [Eq(r.backward, Gzz)]

        op0 = Operator(eqn, subs=grid.spacing_map, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, subs=grid.spacing_map,
                       opt=('advanced', {'openmp': True, 'cire-mingain': 1,
                                         'cire-rotate': rotate}))

        # Check numerical output
        op0(time_M=1)
        summary = op1(time_M=1, r=r1)
        assert np.isclose(norm(r), norm(r1), rtol=1e-5)

        # Check code generation
        assert summary[('section1', None)].ops == exp_ops
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 5
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1

    @switchconfig(profiling='advanced')
    def test_tti_adjoint_akin_v2(self):
        """
        Yet another extrapolation from TTI adjoint which has caused headaches
        in the past.
        """
        so = 12
        to = 2
        fd_order = so // 2

        grid = Grid(shape=(10, 10, 10), dtype=np.float64)
        x, y, z = grid.dimensions

        p = TimeFunction(name='p', grid=grid, space_order=so, time_order=to)
        p1 = TimeFunction(name='p', grid=grid, space_order=so, time_order=to)
        r = TimeFunction(name='r', grid=grid, space_order=so, time_order=to)
        delta = Function(name='delta', grid=grid, space_order=so)
        theta = Function(name='theta', grid=grid, space_order=so)
        phi = Function(name='phi', grid=grid, space_order=so)

        p.data_with_halo[:] = 1.1
        p1.data_with_halo[:] = 1.1
        r.data_with_halo[:] = 0.5
        delta.data_with_halo[:] = 0.2
        theta.data_with_halo[:] = 0.8
        phi.data_with_halo[:] = 0.2

        field = sqrt(1 + 2*delta)*p + r
        Gz = sin(theta) * cos(phi) * field.dx(fd_order=fd_order)
        Gzz = (Gz * cos(theta)).dz(fd_order=fd_order).T
        H0 = field.laplace - Gzz

        eqn = Eq(p.backward, H0)

        op0 = Operator(eqn, subs=grid.spacing_map, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, subs=grid.spacing_map, opt=('advanced', {'openmp': True}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 4
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        self.check_array(arrays[2], ((6, 6), (6, 6), (6, 6)), (xs+12, ys+12, zs+12))
        self.check_array(arrays[3], ((3, 3),), (zs+6,))

        # Check numerical output
        op0(time_M=1)
        summary1 = op1(time_M=1, p=p1)
        exp_p = norm(p)
        assert np.isclose(exp_p, norm(p1), atol=1e-15)

        # Also check against expected operation count to make sure
        # all redundancies have been detected correctly
        assert summary1[('section1', None)].ops == 75

    @pytest.mark.parametrize('rotate', [False, True])
    @switchconfig(profiling='advanced')
    def test_nested_first_derivatives(self, rotate):
        """
        Test that aliasing sub-expressions from nested derivatives aren't split,
        but rather they're captured together and scheduled to a single temporary.
        """
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)
        v1 = TimeFunction(name="v1", grid=grid, space_order=4)

        f.data_with_halo[:] = 0.5
        v.data_with_halo[:] = 1.
        v1.data_with_halo[:] = 1.

        eqn = Eq(v.forward, (v.dx * (1 + 2*f) * f).dx)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-rotate': rotate}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1

        # Check numerical output
        op0(time_M=1)
        summary1 = op1(time_M=1, v=v1)
        expected_v = norm(v)
        assert np.isclose(expected_v, norm(v1), rtol=1e-5)

        # Also check against expected operation count to make sure
        # all redundancies have been detected correctly
        assert summary1[('section0', None)].ops == 14

    def test_undestroyed_preevaluated_derivatives_v1(self):
        grid = Grid(shape=(10, 10))

        v = TimeFunction(name="v", grid=grid, space_order=4)

        expr = -1 * v.dx.dx.evaluate
        eqn = Eq(v.forward, 2 * expr)

        op = Operator(eqn)

        # Check generated code
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 1

    def test_nested_first_derivatives_unbalanced(self):
        grid = Grid(shape=(3, 3))

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=4)

        eq = Eq(u.forward, u.dx.dy + u*(u.dx.dy + 1.))

        op = Operator(eq, opt=('cire-sops', {'cire-mingain': 0}))

        # Make sure there are no undefined symbols
        assert 'dummy' not in str(op)
        op.apply(time_M=0)

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('expr,exp_arrays,exp_ops', [
        ('f.dx.dx + g.dx.dx',
         (1, 2, (0, 1)), (46, 61, 16)),
        ('v.dx.dx + p.dx.dx',
         (2, 2, (0, 2)), (61, 61, 25)),
        ('(v.dx + v.dy).dx - (v.dx + v.dy).dy + 2*f.dx.dx + f*f.dy.dy + f.dx.dx(x0=1)',
         (3, 3, (0, 3)), (217, 201, 74)),
        ('(g*(1 + f)*v.dx).dx + (2*g*f*v.dx).dx',
         (1, 2, (0, 1)), (50, 65, 18)),
        ('g*(f.dx.dx + g.dx.dx)',
         (1, 2, (0, 1)), (47, 62, 17)),
    ])
    def test_sum_of_nested_derivatives(self, expr, exp_arrays, exp_ops):
        """
        Test that aliasing sub-expressions from sums of nested derivatives
        along `x` and `y` are scheduled to *two* different temporaries, not
        three (one per unique derivative argument), thanks to FD linearity.
        """
        grid = Grid(shape=(10, 10, 10), dtype=np.float64)
        x, y, z = grid.dimensions  # noqa

        f = Function(name='f', grid=grid, space_order=4)
        g = Function(name='g', grid=grid, space_order=4)
        p = TimeFunction(name="p", grid=grid, space_order=4, staggered=x)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        f.data_with_halo[:] =\
            np.linspace(-10, 10, f.data_with_halo.size).reshape(*f.shape_with_halo)
        g.data_with_halo[:] =\
            np.linspace(-20, 20, g.data_with_halo.size).reshape(*g.shape_with_halo)
        p.data_with_halo[:] = 0.7
        v.data_with_halo[:] = 1.2

        eqn = Eq(v.forward, eval(expr))

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('collect-derivs', 'cire-sops', {'openmp': True}))
        op2 = Operator(eqn, opt=('cire-sops', {'openmp': True}))
        op3 = Operator(eqn, opt=('advanced', {'openmp': True}))

        # Check code generation
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == exp_arrays[0]
        arrays = [i for i in FindSymbols().visit(op2) if i.is_Array]
        assert len(arrays) == exp_arrays[1]

        bns, pbs = assert_blocking(op3, {'x0_blk0'})

        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        exp_inv, exp_sops = exp_arrays[2]
        assert len(arrays) == exp_inv + exp_sops
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == exp_sops

        # Check numerical output
        op0(time_M=20)
        exp_v = norm(v)
        for n, op in enumerate([op1, op2, op3]):
            v1 = TimeFunction(name="v", grid=grid, space_order=4)
            v1.data_with_halo[:] = 1.2

            summary = op(time_M=20, v=v1)
            assert np.isclose(exp_v, norm(v1), atol=1e-11, rtol=1e-5)

            # Also check against expected operation count to make sure
            # all redundancies have been detected correctly
            for i, exp in enumerate(as_tuple(exp_ops[n])):
                assert summary[('section%d' % i, None)].ops == exp

    def test_derivatives_from_different_levels(self):
        """
        Test catching of derivatives nested at different levels of the
        expression tree.
        """
        grid = Grid(shape=(10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)
        v1 = TimeFunction(name="v", grid=grid, space_order=4)

        f.data_with_halo[:] = 0.5
        v.data_with_halo[:] = 1.2
        v1.data_with_halo[:] = 1.2

        eqn = Eq(v.forward, f*(1. + v).dx + 2.*f*((1. + v).dx + f))

        op = Operator(eqn, opt=('advanced', {'cire-mingain': 0}))

        # Check code generation
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 1

    @pytest.mark.parametrize('rotate', [False, True])
    def test_maxpar_option(self, rotate):
        """
        Test the compiler option `cire-maxpar=True`.
        """
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name="u", grid=grid, space_order=4)

        f.data[:] = 0.0012
        u.data[:] = 1.3
        u1.data[:] = 1.3

        eq = Eq(u.forward, f*u.dy.dy)

        op0 = Operator(eq, opt='noop')
        op1 = Operator(eq, opt=('advanced', {'cire-maxpar': True, 'cire-rotate': rotate}))

        # Check code generation
        bns, _ = assert_blocking(op1, {'x0_blk0'})
        trees = retrieve_iteration_tree(bns['x0_blk0'])
        assert len(trees) == 2
        assert trees[0][1] is trees[1][1]
        assert trees[0][2] is not trees[1][2]

        # Check numerical output
        op0.apply(time_M=2)
        op1.apply(time_M=2, u=u1)
        assert np.isclose(norm(u), norm(u1), rtol=1e-5)

    def test_maxpar_option_v2(self):
        """
        Another test for the compiler option `cire-maxpar=True`.
        """
        grid = Grid(shape=(10, 10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        u = TimeFunction(name='u', grid=grid, space_order=4, save=10)
        u1 = TimeFunction(name="u", grid=grid, space_order=4, save=10)

        f.data[:] = 0.0012
        u.data[0, :] = 1.3
        u1.data[0, :] = 1.3

        eq = Eq(u.forward, f*u.dx.dx)

        op0 = Operator(eq, opt='noop')
        op1 = Operator(eq, opt=('advanced', {'cire-maxpar': True}))

        # Check code generation
        bns, _ = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        self.check_array(arrays[0], ((2, 2), (0, 0), (0, 0)), (xs+4, ys, zs))

        # Check numerical output
        op0.apply(time_M=2)
        op1.apply(time_M=2, u=u1)
        assert np.isclose(norm(u), norm(u1), rtol=1e-10)

    def test_maxpar_option_v3(self):
        """
        Another test for the compiler option `cire-maxpar=True`.
        """
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)
        v = TimeFunction(name="v", grid=grid, space_order=4)

        eq = Eq(u.forward, u.dx.dx + v.dx.dy)

        op = Operator(eq, opt=('advanced', {'cire-maxpar': True}))

        # Check code generation
        xs, ys = self.get_params(op, 'x_size', 'y_size')
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        self.check_array(arrays[0], ((2, 2), (2, 2)), (xs+4, ys+4))
        self.check_array(arrays[1], ((2, 2), (2, 2)), (xs+4, ys+4))
        assert_structure(op, ['t,x,y', 't,x,y'], 't,x,y,x,y')

    @pytest.mark.parametrize('rotate', [False, True])
    def test_blocking_options(self, rotate):
        """
        Test CIRE with all compiler options impacting loop blocking, which in turn
        impact the shape of the created temporaries as well as the surrounding loop
        nests.
        """
        grid = Grid(shape=(20, 20, 20))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=4)
        u1 = TimeFunction(name="u", grid=grid, space_order=4)
        u2 = TimeFunction(name="u", grid=grid, space_order=4)

        f.data_with_halo[:] =\
            np.linspace(-10, 10, f.data_with_halo.size).reshape(*f.shape_with_halo)
        u.data_with_halo[:] =\
            np.linspace(-3, 3, u.data_with_halo.size).reshape(*u.shape_with_halo)
        u1.data_with_halo[:] = u.data_with_halo[:]
        u2.data_with_halo[:] = u.data_with_halo[:]

        eq = Eq(u.forward, u.dx.dx + f*u.dy.dy)

        op0 = Operator(eq, opt='noop')
        op1 = Operator(eq, opt=('advanced', {'blocklevels': 2, 'cire-rotate': rotate,
                                             'min-storage': True}))
        op2 = Operator(eq, opt=('advanced', {'blocklevels': 2, 'par-nested': 0,
                                             'cire-rotate': rotate, 'min-storage': True}))

        # Check code generation
        assert len([i for i in op1.dimensions if i.is_Incr]) == 6 + (2 if rotate else 0)
        if configuration['language'] == 'openmp':
            bns, _ = assert_blocking(op2, {'x0_blk0'})

            pariters = FindNodes(ParallelIteration).visit(bns['x0_blk0'])
            assert len(pariters) == 2

        # Check numerical output
        op0.apply(time_M=2)
        op1.apply(time_M=2, u=u1, x0_blk1_size=2, y0_blk1_size=2)
        op2.apply(time_M=2, u=u2, x0_blk1_size=2, y0_blk1_size=2)
        expected = norm(u)
        assert np.isclose(expected, norm(u1), rtol=1e-5)
        assert np.isclose(expected, norm(u2), rtol=1e-5)

    def test_ftemps_option(self):
        """
        Test the compiler option `cire-ftemps=True`. This will make CIRE use
        TempFunctions, rather than Arrays, to create temporaries, thus giving
        control over allocation and deallocation to the user.
        """
        grid = Grid(shape=(30, 30, 30))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        nthreads = 2
        x0_blk0_size = 8
        y0_blk0_size = 8

        u = TimeFunction(name='u', grid=grid, space_order=3)
        u1 = TimeFunction(name="u", grid=grid, space_order=3)
        u2 = TimeFunction(name="u", grid=grid, space_order=3)
        u3 = TimeFunction(name="u", grid=grid, space_order=3)

        u.data_with_halo[:] = 0.32
        u1.data_with_halo[:] = 0.32
        u2.data_with_halo[:] = 0.32
        u3.data_with_halo[:] = 0.32

        eqn = Eq(u.forward, _R(_R(u[t, x, y, z] + u[t, x+1, y+1, z+1])*3. +
                               _R(u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3.) + 1.)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-mingain': 0,
                                              'cire-ftemps': True}))
        op2 = Operator(eqn, opt=('advanced-fsg', {'openmp': True, 'cire-mingain': 0,
                                                  'cire-ftemps': True}))

        op0(time_M=1, nthreads=nthreads)

        # TempFunctions expect an override
        with pytest.raises(InvalidArgument):
            op1(time_M=1, u=u1)

        # Prepare to run op1
        shape = [nthreads, x0_blk0_size, y0_blk0_size, grid.shape[-1]]
        ofuncs = [i.make(shape) for i in op1.temporaries]
        kwargs = {i.name: i for i in ofuncs}
        # Check numerical output of op1
        op1(time_M=1, u=u1, nthreads=nthreads, **kwargs)
        assert np.allclose(u.data, u1.data, rtol=10e-5)

        # Prepare to run op2
        ofuncs = [i.make(grid.shape) for i in op2.temporaries]
        assert all(i.shape_with_halo == (32, 32, 32) for i in ofuncs)
        kwargs = {i.name: i for i in ofuncs}
        # Check numerical output of op2
        op2(time_M=1, u=u2, **kwargs)
        assert np.allclose(u.data, u2.data, rtol=10e-5)

        # Again op2, but now with automatically derived shape
        ofuncs = [i.make(**u3._arg_values()) for i in op2.temporaries]
        assert all(i.shape_with_halo == (32, 32, 32) for i in ofuncs)
        kwargs = {i.name: i for i in ofuncs}
        # Check numerical output of op2
        op2(time_M=1, u=u3, **kwargs)
        assert np.allclose(u.data, u3.data, rtol=10e-5)

    @pytest.mark.parametrize('rotate', [False, True])
    def test_grouping_fallback(self, rotate):
        """
        MFE for issue #1477.
        """
        space_order = 8
        grid = Grid(shape=(21, 21, 11))

        eps = Function(name='eps', grid=grid, space_order=space_order)
        p = TimeFunction(name='p', grid=grid, time_order=2, space_order=space_order)
        p1 = TimeFunction(name='p', grid=grid, time_order=2, space_order=space_order)

        p.data[:] = 0.02
        p1.data[:] = 0.02
        eps.data_with_halo[:] =\
            np.linspace(0.1, 0.3, eps.data_with_halo.size).reshape(*eps.shape_with_halo)

        eqn = Eq(p.forward, ((1+sqrt(eps)) * p.dy).dy + (p.dz).dz)

        op0 = Operator(eqn, opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, opt=('advanced', {'openmp': True, 'cire-rotate': rotate,
                                              'min-storage': True}))

        # Check code generation
        # `min-storage` leads to one 2D and one 3D Arrays
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = self.get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 3
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        self.check_array(arrays[1], ((4, 4), (0, 0)), (ys+8, zs), rotate)
        self.check_array(arrays[2], ((4, 4),), (zs+8,))  # On purpose w/o `rotate`

        # Check numerical output
        op0.apply(time_M=2)
        op1.apply(time_M=2, p=p1)

        # Note on accuracy:
        # * rtol=1e-7 OK if collapse(3) in op0;
        # * rtol=1e-7 OK if DEVITO_SAFE_MATH=1
        assert np.isclose(norm(p), norm(p1), rtol=1e-6)

    def test_grouping_fallback_v2(self):
        """
        MFE for issue #1586.
        """
        grid = Grid(shape=(20, 20, 20))

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid, space_order=4)

        eqn = Eq(u.forward, (2*f*f*u.dy).dy + (3*f*u.dy).dy)

        op = Operator(eqn, opt=('advanced', {'openmp': False, 'cire-mingain': 4}))

        bns, _ = assert_blocking(op, {'x0_blk0'})

        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 1

    def test_contraction_with_conditional(self):
        """
        MFE for issue #1610.
        """
        grid = Grid(shape=(10, 10))
        time = grid.time_dim
        cond = ConditionalDimension(name='cond', parent=time, condition=time < 5)

        u = TimeFunction(name='u', grid=grid, space_order=4, save=10)

        u.data_with_halo[:] = 1.42

        eqn = Eq(u.forward, u.dy.dy + 1., implicit_dims=[cond])

        op = Operator(eqn, opt=('advanced', {'cire-mingain': 0, 'openmp': True}))

        op.apply(time=8)

        assert len(FindNodes(Conditional).visit(op)) == 1
        assert np.all(u.data[6:] == 1.42)

    def test_collection_from_conditional(self):
        nt = 10
        grid = Grid(shape=(10, 10))
        time_dim = grid.time_dim

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)
        save_shift = Constant(name='save_shift', dtype=np.int32)

        u = TimeFunction(name='u', grid=grid, space_order=4)
        v = TimeFunction(name='v', grid=grid, space_order=4)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)
        vsave = TimeFunction(name='vsave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)

        uss = usave.subs(time_sub, time_sub - save_shift)
        vss = vsave.subs(time_sub, time_sub - save_shift)

        eqn = Eq(u.forward, uss*u.dx.dx + vss*v.dy.dy)
        op = Operator(eqn, opt=('advanced', {'cire-mingain': 1}))
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 2

    def test_hoisting_pow_one(self):
        """
        MFE for issues #1614.
        """
        grid = Grid(shape=(10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        u = TimeFunction(name='u', grid=grid, space_order=4)

        eqn = Eq(u.forward, u*f**1.0)

        op = Operator(eqn)

        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 0


class TestIsoAcoustic(object):

    def run_acoustic_forward(self, opt=None):
        shape = (50, 50, 50)
        spacing = (10., 10., 10.)
        nbl = 10
        nrec = 101
        t0 = 0.0
        tn = 250.0

        # Create two-layer model from preset
        model = demo_model(preset='layers-isotropic', vp_top=3., vp_bottom=4.5,
                           spacing=spacing, shape=shape, nbl=nbl)

        # Source and receiver geometries
        src_coordinates = np.empty((1, len(spacing)))
        src_coordinates[0, :] = np.array(model.domain_size) * .5
        src_coordinates[0, -1] = model.origin[-1] + 2 * spacing[-1]

        rec_coordinates = np.empty((nrec, len(spacing)))
        rec_coordinates[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
        rec_coordinates[:, 1:] = src_coordinates[0, 1:]

        geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates,
                                       t0=t0, tn=tn, src_type='Ricker', f0=0.010)

        solver = AcousticWaveSolver(model, geometry, opt=opt)
        rec, u, summary = solver.forward(save=False)

        op = solver.op_fwd(save=False)

        return u, rec, summary, op

    @switchconfig(profiling='advanced')
    def test_fullopt(self):
        u0, rec0, summary0, op0 = self.run_acoustic_forward(opt=None)
        u1, rec1, summary1, op1 = self.run_acoustic_forward(opt='advanced')

        bns, _ = assert_blocking(op0, {})
        bns, _ = assert_blocking(op1, {'x0_blk0'})  # due to loop blocking

        assert summary0[('section0', None)].ops == 50
        assert summary0[('section1', None)].ops == 139
        assert np.isclose(summary0[('section0', None)].oi, 2.851, atol=0.001)

        assert summary1[('section0', None)].ops == 31
        assert np.isclose(summary1[('section0', None)].oi, 1.767, atol=0.001)

        assert np.allclose(u0.data, u1.data, atol=10e-5)
        assert np.allclose(rec0.data, rec1.data, atol=10e-5)


class TestTTI(object):

    @cached_property
    def model(self):
        # TTI layered model for the tti test, no need for a smooth interace
        # bewtween the two layer as the compilation passes are tested, not the
        # physical prettiness of the result -- which ultimately saves time
        return demo_model('layers-tti', nlayers=3, nbl=10, space_order=4,
                          shape=(50, 50, 50), spacing=(20., 20., 20.), smooth=False)

    @cached_property
    def geometry(self):
        nrec = 101
        t0 = 0.0
        tn = 250.

        # Source and receiver geometries
        src_coordinates = np.empty((1, len(self.model.spacing)))
        src_coordinates[0, :] = np.array(self.model.domain_size) * .5
        src_coordinates[0, -1] = self.model.origin[-1] + 2 * self.model.spacing[-1]

        rec_coordinates = np.empty((nrec, len(self.model.spacing)))
        rec_coordinates[:, 0] = np.linspace(0., self.model.domain_size[0], num=nrec)
        rec_coordinates[:, 1:] = src_coordinates[0, 1:]

        geometry = AcquisitionGeometry(self.model, rec_coordinates, src_coordinates,
                                       t0=t0, tn=tn, src_type='Gabor', f0=0.010)
        return geometry

    def tti_operator(self, opt, space_order=4):
        return AnisotropicWaveSolver(self.model, self.geometry,
                                     space_order=space_order, opt=opt)

    @cached_property
    def tti_noopt(self):
        wavesolver = self.tti_operator(opt=None)
        rec, u, v, summary = wavesolver.forward()

        # Make sure no opts were applied
        op = wavesolver.op_fwd(False)
        assert len(op._func_table) == 0
        assert summary[('section0', None)].ops == 737

        return v, rec

    @switchconfig(profiling='advanced')
    def test_fullopt(self):
        wavesolver = self.tti_operator(opt='advanced')
        rec, u, v, summary = wavesolver.forward()

        assert np.allclose(self.tti_noopt[0].data, v.data, atol=10e-1)
        assert np.allclose(self.tti_noopt[1].data, rec.data, atol=10e-1)

        # Check expected opcount/oi
        assert summary[('section1', None)].ops == 90
        assert np.isclose(summary[('section1', None)].oi, 2.031, atol=0.001)

        # With optimizations enabled, there should be exactly four BlockDimensions
        op = wavesolver.op_fwd()
        block_dims = [i for i in op.dimensions if i.is_Block]
        assert len(block_dims) == 4
        x, x0_blk0, y, y0_blk0 = block_dims
        assert x.parent is x0_blk0
        assert y.parent is y0_blk0
        assert not x._defines & y._defines

        # Also, in this operator, we expect seven temporary Arrays:
        # * all of the seven Arrays are allocated on the heap
        # * with OpenMP, five Arrays are defined globally, and two additional
        #   Arrays are defined locally
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        extra_arrays = 2
        assert len(arrays) == 4 + extra_arrays
        assert all(i._mem_heap and not i._mem_external for i in arrays)
        bns, pbs = assert_blocking(op, {'x0_blk0'})

        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 6
        assert all(not i._mem_external for i in arrays)
        assert len([i for i in arrays if i._mem_heap]) == 6
        vexpanded = 2 if configuration['language'] == 'openmp' else 0
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == vexpanded

    @skipif(['nompi'])
    @switchconfig(profiling='advanced')
    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_fullopt_w_mpi(self):
        tti_noopt = self.tti_operator(opt=None)
        rec0, u0, v0, _ = tti_noopt.forward()
        tti_agg = self.tti_operator(opt='advanced')
        rec1, u1, v1, _ = tti_agg.forward()

        assert np.allclose(v0.data, v1.data, atol=10e-1)
        assert np.allclose(rec0.data, rec1.data, atol=10e-1)

        # Run a quick check to be sure MPI-full-mode code was actually generated
        op = tti_agg.op_fwd(False)
        assert len(op._func_table) == 7
        assert 'pokempi0' in op._func_table

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,expected', [
        (8, 152), (16, 270)
    ])
    def test_opcounts(self, space_order, expected):
        op = self.tti_operator(opt='advanced', space_order=space_order)
        sections = list(op.op_fwd()._profiler._sections.values())
        assert sections[1].sops == expected

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,expected', [
        (4, 111),
    ])
    def test_opcounts_adjoint(self, space_order, expected):
        wavesolver = self.tti_operator(opt=('advanced', {'openmp': False}))
        op = wavesolver.op_adj()

        assert op._profiler._sections['section1'].sops == expected
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 7


class TestTTIv2(object):

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,expected', [
        (4, 191), (12, 383)
    ])
    def test_opcounts(self, space_order, expected):
        grid = Grid(shape=(3, 3, 3))

        s = 0.00067
        u = TimeFunction(name='u', grid=grid, space_order=space_order)
        v = TimeFunction(name='v', grid=grid, space_order=space_order)
        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=grid)
        m = Function(name='m', grid=grid)
        e = Function(name='e', grid=grid)
        d = Function(name='d', grid=grid)

        ang0 = cos(f)
        ang1 = sin(f)
        ang2 = cos(g)
        ang3 = sin(g)

        H1u = (ang1*ang1*ang2*ang2*u.dx2 +
               ang1*ang1*ang3*ang3*u.dy2 +
               ang0*ang0*u.dz2 +
               2*ang1*ang1*ang3*ang2*u.dxdy +
               2*ang0*ang1*ang3*u.dydz +
               2*ang0*ang1*ang2*u.dxdz)
        H2u = -H1u + u.laplace

        H1v = (ang1*ang1*ang2*ang2*v.dx2 +
               ang1*ang1*ang3*ang3*v.dy2 +
               ang0*ang0*v.dz2 +
               2*ang1*ang1*ang3*ang2*v.dxdy +
               2*ang0*ang1*ang3*v.dydz +
               2*ang0*ang1*ang2*v.dxdz)
        H2v = -H1v + v.laplace

        eqns = [Eq(u.forward, (2*u - u.backward) + s**2/m * (e * H2u + H1v)),
                Eq(v.forward, (2*v - v.backward) + s**2/m * (d * H2v + H1v))]
        op = Operator(eqns, openmp=True)

        # Check code generation
        _, pbs = assert_blocking(op, {'x0_blk0'})
        arrays = FindNodes(VExpanded).visit(pbs['x0_blk0'])
        assert len(arrays) == 4
        assert all(len(i.pointee.shape) == 2 for i in arrays)  # Expected 2D arrays
        sections = list(op._profiler._sections.values())
        assert len(sections) == 2
        assert sections[0].sops == 4
        assert sections[1].sops == expected
