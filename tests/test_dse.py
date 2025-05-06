from functools import cached_property

import numpy as np
import pytest

from sympy import Mul  # noqa

from conftest import (skipif, EVAL, _R, assert_structure, assert_blocking,  # noqa
                      get_params, get_arrays, check_array)
from devito import (NODE, Eq, Inc, Constant, Function, TimeFunction,  # noqa
                    SparseTimeFunction, Dimension, SubDimension,
                    ConditionalDimension, DefaultDimension, Grid, Operator,
                    norm, grad, div, dimensions, switchconfig, configuration,
                    first_derivative, solve, transpose, Abs, cos, exp,
                    sin, sqrt, floor, Ge, Lt, Derivative)
from devito.exceptions import InvalidArgument, InvalidOperator
from devito.ir import (Conditional, DummyEq, Expression, Iteration, FindNodes,
                       FindSymbols, ParallelIteration, retrieve_iteration_tree)
from devito.passes.clusters.aliases import collect
from devito.passes.clusters.factorization import collect_nested
from devito.passes.iet.parpragma import VExpanded
from devito.symbolics import (INT, FLOAT, DefFunction, FieldFromPointer,  # noqa
                              IndexedPointer, Keyword, SizeOf, estimate_cost,
                              pow_to_mul, indexify)
from devito.tools import as_tuple
from devito.types import Scalar, Symbol, PrecomputedSparseTimeFunction

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
    ('fa[x]**s', 'fa[x]**s'),
    ('fa[x]**(-s)', 'fa[x]**(-s)'),
    ('-2/(s**2)', '-2/(s*s)'),
    ('-fa[x]', '-fa[x]'),
    ('Mul(SizeOf("char"), '
     '-IndexedPointer(FieldFromPointer("size", fa._C_symbol), x), evaluate=False)',
     'sizeof(char)*(-fa_vec->size[x])'),
    ('sqrt(fa[x]**4)', 'sqrt(fa[x]*fa[x]*fa[x]*fa[x])'),
    ('sqrt(fa[x])**2', 'fa[x]'),
    ('fa[x]**-2', '1/(fa[x]*fa[x])'),
    ('cos(fa[x]*fa[x])*cos(fa[x]*fa[x])', 'cos(fa[x]*fa[x])*cos(fa[x]*fa[x])'),
])
def test_pow_to_mul(expr, expected):
    grid = Grid((4, 5))
    x, y = grid.dimensions

    s = Scalar(name='s')  # noqa
    fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
    fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa

    assert str(pow_to_mul(eval(expr))) == expected


@pytest.mark.parametrize('expr,expected', [
    ('s - SizeOf("int")*fa[x]', 's - sizeof(int)*fa[x]'),
    ('foo(4*fa[x] + 4*fb[x])', 'foo(4*(fa[x] + fb[x]))'),
    ('floor(0.1*a + 0.1*fa[x])', 'floor(0.1*(a + fa[x]))'),
    ('floor(0.1*(a + fa[x]))', 'floor(0.1*(a + fa[x]))'),
])
def test_factorize(expr, expected):
    grid = Grid((4, 5))
    x, y = grid.dimensions

    s = Scalar(name='s', dtype=np.float32)  # noqa
    a = Symbol(name='a', dtype=np.float32)  # noqa
    fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
    fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa
    foo = lambda *args: DefFunction('foo', tuple(args))  # noqa

    assert str(collect_nested(eval(expr))) == expected


@pytest.mark.parametrize('expr,expected,estimate', [
    ('Eq(t0, 3)', 0, False),
    ('Eq(t0, 4.5)', 0, False),
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
    ('Eq(t0, Abs(t1 + t2))', 2, False),
    ('Eq(t0, Abs(t1 + t2))', 6, True),
    # Integer arithmetic should not count
    ('Eq(t0, INT(t1))', 0, True),
    ('Eq(t0, INT(t1*t0))', 1, True),
    ('Eq(t0, 2 + INT(t1*t0))', 1, True),
    ('Eq(t0, FLOAT(t1))', 0, True),
    ('Eq(t0, FLOAT(t1*t2*t3))', 2, True),
    ('Eq(t0, 1 + FLOAT(t1*t2*t3))', 3, True),  # The 1 gets casted to float
    ('Eq(t0, 1 + t3)', 0, False),
    ('Eq(t0, 1 + t3)', 0, True),
    ('Eq(t0, t3 + t4)', 0, True),
    ('Eq(t0, 2*t3)', 0, True),
    ('Eq(t0, 2*t1)', 1, True),
    ('Eq(t0, -4 + INT(t3*t4 + t3))', 0, True),
    # Custom routines and types
    ('Eq(t0, SizeOf("int"))', 0, True),
    ('Eq(t0, SizeOf("int"))', 0, False),
    ('Eq(t0, foo(k, 9))', 0, False),
    ('Eq(t0, foo(k, 9))', 0, True),
    ('Eq(t0, foo(k, t0 + 9))', 1, True),
    ('Eq(t0, foo(k, cos(t0 + 9)))', 101, True),
    ('Eq(t0, ffp)', 0, True),
    ('Eq(t0, ffp + 1.)', 1, True),
    ('Eq(t0, ffp + ffp)', 1, True),
    # W/ StencilDimensions
    ('Eq(fb, fd.dx)', 10, False),
    ('Eq(fb, fd.dx)', 10, True),
    ('Eq(fb, fd.dx._evaluate(expand=False))', 10, False),
    ('Eq(fb, fd.dx.dy + fa.dx)', 65, False),
    # Ensure redundancies aren't counted
    ('Eq(t0, fd.dx.dy + fa*fd.dx.dy)', 62, True),
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
    t3 = Scalar(name='t3', dtype=np.int32)  # noqa
    t4 = Scalar(name='t4', dtype=np.int32)  # noqa
    fa = Function(name='fa', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fb = Function(name='fb', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fc = Function(name='fc', grid=grid)  # noqa
    fd = Function(name='fd', grid=grid, space_order=4)  # noqa
    foo = lambda *args: DefFunction('foo', tuple(args))  # noqa
    k = Keyword('k')  # noqa
    ffp = FieldFromPointer('size', fa._C_symbol)  # noqa

    assert estimate_cost(eval(expr), estimate) == expected


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


class TestLifting:

    @pytest.mark.parametrize('exprs,expected', [
        # none (different distance)
        (['Eq(y.symbolic_max, g[0, x], implicit_dims=(t, x))',
         'Inc(h1[0, 0], 1, implicit_dims=(t, x, y))'],
         [6., 0., 0.]),
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

    def test_implicit_only(self):
        grid = Grid(shape=(5, 5))
        time = grid.time_dim
        u = TimeFunction(name="u", grid=grid, time_order=1)
        idimeq = Eq(Symbol('s'), 1, implicit_dims=time)

        op = Operator([Eq(u.forward, u + 1.), idimeq])
        trees = retrieve_iteration_tree(op)

        assert len(trees) == 2
        assert_structure(op, ['t,x,y', 't'], 'txy')
        assert trees[1].dimensions == [time]

    def test_scalar_cond(self):
        grid = Grid(shape=(5, 5))
        time = grid.time_dim
        u = TimeFunction(name="u", grid=grid, time_order=1)
        bt = ConditionalDimension(name="bt", parent=time, condition=Ge(time, 2))

        W = (1 - exp(-(time - 5)/5))
        eqns = [Eq(u.forward, 1),
                Eq(u.forward, u.forward * (1 - W) + W * u, implicit_dims=bt)]
        op = Operator(eqns)

        trees = retrieve_iteration_tree(op)

        assert len(trees) == 3
        assert_structure(op, ['t', 't,x,y', 't,x,y'], 'txyxy')
        assert trees[0].dimensions == [time]


class TestAliases:

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
        aliases.filter(lambda a: a.score > 0)

        assert len(aliases) == len(expected)
        assert all(i.pivot in expected for i in aliases)

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+2), rotate)

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
        ys, zs = get_params(op1, 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        check_array(arrays[0], ((1, 1), (1, 1)), (ys+2, zs+2), rotate)

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        check_array(arrays[0], ((1, 1), (1, 1), (0, 0)), (xs+2, ys+2, zs), rotate)

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
        xs, ys, zs = get_params(op1, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 1
        check_array(arrays[0], ((0, 0), (0, 0), (1, 0)), (xs, ys, zs+1))

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
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+2), rotate)

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        check_array(arrays[0], ((1, 0), (1, 0), (0, 0)), (xs+1, ys+1, zs), rotate)
        check_array(arrays[1], ((1, 1), (0, 0)), (ys+2, zs), rotate)

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
        xs, ys, zs = get_params(op1, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(op1)) == 1
        check_array(arrays[0], ((2, 2), (0, 0), (0, 0)), (xs+4, ys, zs))
        check_array(arrays[1], ((2, 2), (0, 0)), (ys+4, zs))

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
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        check_array(arrays[0], ((1, 0), (1, 0), (0, 0)), (xs+1, ys+1, zs), rotate)
        check_array(arrays[1], ((1, 1), (1, 0)), (ys+2, zs+1), rotate)

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 2
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        check_array(arrays[0], ((1, 0), (1, 1), (0, 0)), (xs+1, ys+2, zs), rotate)
        check_array(arrays[1], ((1, 0), (1, 1), (0, 0)), (xs+1, ys+2, zs), rotate)

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 3
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3
        check_array(arrays[0], ((1, 0), (1, 0)), (xs+1, zs+1), rotate)
        check_array(arrays[1], ((1, 1), (1, 1)), (ys+2, zs+2), rotate)
        check_array(arrays[2], ((1, 1), (1, 1)), (ys+2, zs+2), rotate)

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
        ys, zs = get_params(op1, 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 1
        check_array(arrays[0], ((1, 1), (1, 0)), (ys+2, zs+1), rotate)

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

        op = Operator(eq, opt=('advanced', {'openmp': False}))

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
        assert len(vexpandeds) == (2 if 'openmp' in configuration['language'] else 0)
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
        assert sum(i.ops for i in summary1.values()) == 74

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
        assert len(trees) == 4 if rotate else 2
        assert trees[-2][-1].nodes[0].body[0].write.is_Array
        assert trees[-1][-1].nodes[0].body[0].write is u
        trees = retrieve_iteration_tree(bns['x1_blk0'])
        assert len(trees) == 4 if rotate else 2
        assert trees[-2][-1].nodes[0].body[0].write.is_Array
        assert trees[-1][-1].nodes[0].body[0].write is v

        # Check numerical output
        op0(time_M=1)
        op1(time_M=1, u=u1, v=v1)
        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

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
        if op._options['linearize']:
            exprs = exprs[6:]
        sqrt_exprs = exprs[:2]
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
        assert_structure(op, ['t,x,y', 't,f', 't,f,x,y'], 't,x,y,f,f,x,y')

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
            ys = get_params(op, 'y_size')[0]
            arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
            assert len(arrays) == 1
            check_array(arrays[0], ((0, 0),), (ys,))
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

        xs, ys, zs = get_params(op, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 3
        check_array(arrays[0], ((0, 0),), (ys,))
        check_array(arrays[1], ((0, 0), (0, 0)), (xs, zs))
        check_array(arrays[2], ((0, 0), (0, 0)), (xs, ys))

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

        xs, ys, zs = get_params(op, 'x_size', 'y_size', 'z_size')
        arrays = get_arrays(op)
        assert len(arrays) == 1
        check_array(arrays[0], ((1, 0), (1, 0), (0, 0)), (xs+1, ys+1, zs))
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

        u = TimeFunction(name="u", grid=grid, space_order=8, time_order=2)

        pde = u.dt2 - (u.dx.dx + u.dy.dy) + u.dx.dy
        eq = Eq(u.forward, solve(pde, u.forward))

        op = Operator(eq)
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 2
        assert op._profiler._sections['section0'].sops == 67

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

    def test_hoisting_symbolic_divs(self):
        grid = Grid(shape=(3, 3))

        f = Function(name='f', grid=grid, space_order=4)
        s0 = Scalar(name='s0')
        s1 = Scalar(name='s1')

        eq = Eq(f, f*(s0**-s1))

        op = Operator(eq)

        assert op._profiler._sections['section0'].sops == 1
        assert str(op.body.body[-1].body[0].body[0].expr.rhs) == str(s0**-s1)

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
        bns, _ = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = get_params(op1, 'x_size', 'y_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        check_array(arrays[0], ((1, 1), (1, 1), (1, 1)), (xs+2, ys+2, zs+2))

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

        f = Function(name='f', grid=grid, space_order=so, parameter=True)
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

    @pytest.mark.parametrize('so_ops', [(4, 113)])
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

        op = Operator(eqn, subs=grid.spacing_map, opt=('advanced', {'openmp': True}))

        # Check code generation
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        assert op._profiler._sections['section1'].sops == exp_ops
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 5
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3
        xs, ys, zs = get_params(op, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        # The three kind of derivatives taken -- in x, y, z -- all are over
        # different expressions, so this leads to three temporaries of dimensionality,
        # in particular 3D for the x-derivative, 2D for the y-derivative, and 1D
        # for the z-derivative
        check_array(arrays[2], ((2, 1), (0, 0), (0, 0)), (xs+3, ys, zs))
        check_array(arrays[3], ((2, 1), (0, 0)), (ys+3, zs))
        check_array(arrays[4], ((2, 1),), (zs+3,))

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

        op = Operator(eqn, subs=grid.spacing_map, opt=('advanced', {'openmp': True}))

        # Check code generation
        assert op._profiler._sections['section1'].sops == exp_ops
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 6
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3

    @pytest.mark.parametrize('so_ops', [(4, 49)])
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

        op = Operator(eqn, subs=grid.spacing_map, opt=('advanced', {'openmp': True}))

        # Check code generation
        assert op._profiler._sections['section1'].sops == exp_ops
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        assert len([i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]) == 7
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 3

    @pytest.mark.parametrize('so_ops', [(4, 147), (8, 211)])
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

        op = Operator(eqns, subs=grid.spacing_map, opt=('advanced', {'openmp': True}))

        # Check code generation
        assert op._profiler._sections['section1'].sops == exp_ops
        bns, pbs = assert_blocking(op, {'x0_blk0'})
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 10
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 6

    @pytest.mark.parametrize('so_ops', [(4, 31), (8, 69)])
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

        op0 = Operator(eqn, subs=grid.spacing_map,
                       opt=('noop', {'openmp': True}))
        op1 = Operator(eqn, subs=grid.spacing_map,
                       opt=('advanced', {'openmp': True,
                                         'cire-schedule': 0}))

        # Check code generation
        bns, pbs = assert_blocking(op1, {'x0_blk0'})
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 4
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        check_array(arrays[2], ((6, 6), (6, 6), (6, 6)), (xs+12, ys+12, zs+12))
        check_array(arrays[3], ((3, 3),), (zs+6,))

        # Check numerical output
        op0(time_M=1)
        summary1 = op1(time_M=1, p=p1)
        exp_p = norm(p)
        assert np.isclose(exp_p, norm(p1), atol=1e-15)

        # Also check against expected operation count to make sure
        # all redundancies have been detected correctly
        assert summary1[('section1', None)].ops == 75

    @switchconfig(profiling='advanced')
    def test_tti_adjoint_akin_v3(self):
        so = 8
        fd_order = 2

        grid = Grid(shape=(20, 20, 20))
        x, y, z = grid.dimensions

        vx = TimeFunction(name="vx", grid=grid, space_order=so)
        vy = TimeFunction(name="vy", grid=grid, space_order=so)
        vz = TimeFunction(name="vz", grid=grid, space_order=so)
        txy = TimeFunction(name="txy", grid=grid, space_order=so)
        txz = TimeFunction(name="txz", grid=grid, space_order=so)
        theta = Function(name='theta', grid=grid, space_order=so)
        phi = Function(name='phi', grid=grid, space_order=so)

        r00 = cos(theta)*cos(phi)
        r01 = cos(theta)*sin(phi)
        r02 = -sin(theta)
        r10 = -sin(phi)
        r11 = cos(phi)
        r12 = cos(theta)
        r20 = sin(theta)*cos(phi)
        r21 = sin(theta)*sin(phi)
        r22 = cos(theta)

        def foo0(field):
            return ((r00 * field).dx(x0=x+x.spacing/2) +
                    Derivative(r01 * field, x, deriv_order=0, fd_order=fd_order,
                               x0=x+x.spacing/2).dy(x0=y) +
                    Derivative(r02 * field, x, deriv_order=0, fd_order=fd_order,
                               x0=x+x.spacing/2).dz(x0=z))

        def foo1(field):
            return (Derivative(r10 * field, y, deriv_order=0, fd_order=fd_order,
                               x0=y+y.spacing/2).dx(x0=x) +
                    (r11 * field).dy(x0=y+y.spacing/2) +
                    Derivative(r12 * field, y, deriv_order=0, fd_order=fd_order,
                               x0=y+y.spacing/2).dz(x0=z))

        def foo2(field):
            return (Derivative(r20 * field, z, deriv_order=0, fd_order=fd_order,
                               x0=z+z.spacing/2).dx(x0=x) +
                    Derivative(r21 * field, z, deriv_order=0, fd_order=fd_order,
                               x0=z+z.spacing/2).dy(x0=y) +
                    (r22 * field).dz(x0=z+z.spacing/2))

        eqns = [Eq(txz.forward, txz + foo0(vz.forward) + foo2(vx.forward)),
                Eq(txy.forward, txy + foo0(vy.forward) + foo1(vx.forward))]

        op = Operator(eqns, subs=grid.spacing_map,
                      opt=('advanced', {'openmp': True,
                                        'cire-rotate': True}))

        # Check code generation
        bns, _ = assert_blocking(op, {'x0_blk0'})
        xs, ys, zs = get_params(op, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 10
        assert len([i for i in arrays if i.shape == (zs,)]) == 2
        assert len([i for i in arrays if i.shape == (9, zs)]) == 2

        assert op._profiler._sections['section1'].sops == 184

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
        assert summary1[('section0', None)].ops == 16

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
         (3, 3, (0, 3)), (218, 202, 66)),
        ('(g*(1 + f)*v.dx).dx + (2*g*f*v.dx).dx',
         (1, 1, (0, 1)), (52, 66, 20)),
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
            for i, expected in enumerate(as_tuple(exp_ops[n])):
                assert summary[('section%d' % i, None)].ops == expected

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

        op = Operator(eqn, opt=('advanced', {'cire-mingain': 0,
                                             'cire-schedule': 0}))

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
        if rotate:
            assert len(trees) == 5
        else:
            assert len(trees) == 2
            assert trees[0][2] is not trees[1][2]
        assert trees[0][1] is trees[1][1]

        # Check numerical output
        op0.apply(time_M=2)
        op1.apply(time_M=2, u=u1)
        assert np.isclose(norm(u), norm(u1), rtol=1e-5)

    def test_multiple_rotating_dims(self):
        space_order = 8
        grid = Grid(shape=(51, 51, 51))
        x, y, z = grid.dimensions

        dt = 0.1
        nt = 5

        u = TimeFunction(name="u", grid=grid, space_order=space_order)
        vx = TimeFunction(name="vx", grid=grid, space_order=space_order)
        vy = TimeFunction(name="vy", grid=grid, space_order=space_order)

        f = Function(name='f', grid=grid, space_order=space_order)
        g = Function(name='g', grid=grid, space_order=space_order)

        expr0 = 1-cos(f)**2
        expr1 = sin(f)*cos(f)
        expr2 = sin(g)*cos(f)
        expr3 = (1-cos(g))*sin(f)*cos(f)

        stencil0 = ((expr0*vx.forward).dx(x0=x-x.spacing/2) +
                    Derivative(expr1*vx.forward, x, deriv_order=0, fd_order=2,
                               x0=x-x.spacing/2).dy(x0=y) +
                    Derivative(expr2*vx.forward, x, deriv_order=0, fd_order=2,
                               x0=x-x.spacing/2).dz(x0=z))
        stencil1 = Derivative(expr3*vy.forward, y, deriv_order=0, fd_order=2,
                              x0=y-y.spacing/2).dx(x0=x)

        eqns = [Eq(vx.forward, u*.1),
                Eq(vy.forward, u*.1),
                Eq(u.forward, stencil0 + stencil1 + .1)]

        op0 = Operator(eqns)
        op1 = Operator(eqns, opt=("advanced", {"cire-rotate": True}))

        f.data_with_halo[:] = .3
        g.data_with_halo[:] = .7

        u1 = u.func(name='u1')
        vx1 = vx.func(name='vx1')
        vy1 = vy.func(name='vy1')

        op0.apply(time_m=0, time_M=nt-2, dt=dt)

        # NOTE: the main issue leading to this test was actually failing
        # to jit-compile `op1`. However, we also check numerical correctness
        op1.apply(time_m=0, time_M=nt-2, dt=dt, u=u1, vx=vx1, vy=vy1)

        assert np.allclose(u.data, u1.data, rtol=1e-3)

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 1
        check_array(arrays[0], ((2, 2), (0, 0), (0, 0)), (xs+4, ys, zs))

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
        xs, ys = get_params(op, 'x_size', 'y_size')
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        check_array(arrays[0], ((2, 2), (2, 2)), (xs+4, ys+4))
        check_array(arrays[1], ((2, 2), (2, 2)), (xs+4, ys+4))
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
                                             'linearize': False,
                                             'min-storage': True}))
        op2 = Operator(eq, opt=('advanced', {'blocklevels': 2, 'par-nested': 0,
                                             'linearize': False,
                                             'cire-rotate': rotate, 'min-storage': True}))

        # Check code generation
        if 'openmp' in configuration['language']:
            prefix = ['t']
        else:
            prefix = []
        if rotate:
            assert_structure(
                op1,
                prefix + ['t,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x',
                          't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,xc',
                          't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,xc,y,z',
                          't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,y',
                          't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,y,yc',
                          't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,y,yc,z',
                          't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,y,z'],
                't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,xc,y,z,y,yc,z,z'
            )
        else:
            assert_structure(
                op1,
                prefix + ['t,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,y,z']*3,
                't,x0_blk0,y0_blk0,x0_blk1,y0_blk1,x,y,z,x,y,z,y,z'
            )
        if 'openmp' in configuration['language']:
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
                                              'cire-ftemps': True,
                                              'linearize': False}))
        op2 = Operator(eqn, opt=('advanced-fsg', {'openmp': True, 'cire-mingain': 0,
                                                  'cire-ftemps': True}))

        op0(time_M=1, nthreads=nthreads)

        # TempFunctions expect an override
        with pytest.raises(InvalidArgument):
            op1(time_M=1, u=u1)

        block_dims = [i for i in op1.dimensions if i.is_Block and i._depth == 1]
        assert len(block_dims) == 2
        mapper = {d.root: d for d in block_dims}
        x0_blk0_size = mapper[x]._arg_defaults()[mapper[x].step.name]
        y0_blk0_size = mapper[y]._arg_defaults()[mapper[y].step.name]

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
        xs, ys, zs = get_params(op1, 'x0_blk0_size', 'y0_blk0_size', 'z_size')
        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 3
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == 2
        check_array(arrays[1], ((4, 4), (0, 0)), (ys+8, zs), rotate)
        check_array(arrays[2], ((4, 4),), (zs+8,))  # On purpose w/o `rotate`

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

    def test_invariants_with_conditional(self):
        factor = 4
        grid = Grid(shape=(10, 10))
        x, y = grid.dimensions
        time_dim = grid.time_dim

        fd = DefaultDimension(name='fd', default_value=2)
        f = Function(name='f', dimensions=(fd,), shape=(2,))
        g = Function(name='g', grid=grid)

        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)
        u = TimeFunction(name='u', grid=grid, space_order=2)
        uf = Function(name='uf', grid=grid, space_order=2, dimensions=(fd, x, y),
                      shape=(2, 10, 10))

        # Standard case
        eqn = Eq(u, u - (cos(time_sub * factor * f) * uf))

        op = Operator(eqn, opt='advanced')
        assert_structure(op, ['t', 't,fd', 't,fd,x,y'], 't,fd,x,y')
        # Make sure it compiles
        op.cfunction

        # Check hoisting for time invariant
        eqn = Eq(u, u - (cos(time_sub * factor * f) * sin(g) * uf))

        op = Operator(eqn, opt='advanced')
        assert_structure(op, ['x,y', 't', 't,fd', 't,fd,x,y'], 'x,y,t,fd,x,y')
        # Make sure it compiles
        op.cfunction

    def test_hoisting_pow_one(self):
        """
        MFE for issue #1614.
        """
        grid = Grid(shape=(10, 10))

        f = Function(name='f', grid=grid, space_order=4)
        u = TimeFunction(name='u', grid=grid, space_order=4)

        eqn = Eq(u.forward, u*f**1.0)

        op = Operator(eqn)

        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 0

    @pytest.mark.parametrize('expr,expected', [
        ('u.dy.dy', False),  # No SEPARABLE as hyperplane degenerates to 1D space
        ('u.dx.dx + u.dy.dy', True),
        ('u.dx.dx + u.dx.dy', True)
    ])
    def test_separable_property(self, expr, expected):
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)

        eq = Eq(u.forward, eval(expr))

        if expected:
            with pytest.raises(NotImplementedError):
                Operator(eq, opt=('cire-sops', 'opt-hyperplanes'))
        else:
            Operator(eq, opt=('cire-sops', 'opt-hyperplanes'))

    def test_premature_evalderiv_lowering(self):
        """
        MFE for issue #1978.
        """
        grid = Grid(shape=(10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=4)

        # Not really a custom derivative, but the enforced pre-evaluation makes
        # it behaves as if it were one
        mock_custom_deriv = u.dx.dy.evaluate

        # This symbolic operation -- creating an Add between an arbitray object
        # and an EvalDerivative -- caused the EvalDerivative to be prematurely
        # simplified being flatten into an Add
        expr0 = u.dt - mock_custom_deriv
        expr = -expr0

        eq = Eq(u.forward, expr)

        op = Operator(eq)

        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == 1
        assert op._profiler._sections['section0'].sops == 16

    def test_issue_2163(self):
        grid = Grid((3, 3))
        z = grid.dimensions[-1]
        mapper = {z: INT(abs(z-1))}

        u = TimeFunction(name="u", grid=grid)
        op = Operator(Eq(u.forward, u.dy.dy.subs(mapper),
                         subdomain=grid.interior))
        assert_structure(op, ['t,x,y'], 'txy')

    def test_dtype_aliases(self):
        a = np.arange(64).reshape((8, 8))
        grid = Grid(shape=a.shape, extent=(7, 7))

        so = 2
        f = Function(name='f', grid=grid, space_order=so, dtype=np.int32)
        f.data[:] = a

        fo = Function(name='fo', grid=grid, space_order=so, dtype=np.int32)
        op = Operator(Eq(fo, f.dx))
        op.apply()

        k = 2 if op._options['linearize'] else 0
        assert FindNodes(Expression).visit(op)[k].dtype == np.float32
        assert np.all(fo.data[:-1, :-1] == 8)

    def test_sparse_const(self):
        grid = Grid((11, 11, 11))

        u = TimeFunction(name="u", grid=grid)
        src = PrecomputedSparseTimeFunction(name="src", grid=grid, npoint=1, nt=11,
                                            r=2, interpolation_coeffs=np.ones((1, 3, 2)),
                                            gridpoints=[[5, 5, 5]])
        u.data.fill(1.)

        op = Operator(src.interpolate(u))

        cond = FindNodes(Conditional).visit(op)
        assert len(cond) == 1
        assert len(cond[0].args['then_body'][0].exprs) == 1
        assert all(e.is_scalar for e in cond[0].args['then_body'][0].exprs)

        op()
        assert np.all(src.data == 8)


class TestIsoAcoustic:

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
        assert summary0[('section1', None)].ops == 44
        assert np.isclose(summary0[('section0', None)].oi, 2.851, atol=0.001)

        assert summary1[('section0', None)].ops == 31
        assert summary1[('section1', None)].ops == 88
        assert summary1[('section2', None)].ops == 25
        assert np.isclose(summary1[('section0', None)].oi, 1.767, atol=0.001)

        assert np.allclose(u0.data, u1.data, atol=10e-5)
        assert np.allclose(rec0.data, rec1.data, atol=10e-5)


class TestTTI:

    @cached_property
    def model(self):
        # TTI layered model for the tti test, no need for a smooth interace
        # bewtween the two layer as the compilation passes are tested, not the
        # physical prettiness of the result -- which ultimately saves time
        return demo_model('layers-tti', nlayers=3, nbl=10, space_order=8,
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
        assert summary[('section0', None)].ops == 743

        return v, rec

    @switchconfig(profiling='advanced')
    def test_fullopt(self):
        wavesolver = self.tti_operator(opt='advanced')
        rec, u, v, summary = wavesolver.forward()

        assert np.allclose(self.tti_noopt[0].data, v.data, atol=10e-1)
        assert np.allclose(self.tti_noopt[1].data, rec.data, atol=10e-1)

        # Check expected opcount/oi
        assert summary[('section1', None)].ops == 92
        assert np.isclose(summary[('section1', None)].oi, 1.99, atol=0.001)

        # With optimizations enabled, there should be exactly four BlockDimensions
        op = wavesolver.op_fwd()
        block_dims = [i for i in op.dimensions if i.is_Block]
        assert len(block_dims) == 4
        x, x0_blk0, y, y0_blk0 = block_dims
        assert x.parent is x0_blk0
        assert y.parent is y0_blk0
        assert not x._defines & y._defines

        # Also, in this operator, we expect six temporary Arrays:
        # * all of the six Arrays are allocated on the heap
        # * with OpenMP:
        #   four Arrays are defined globally for the cos/sin temporaries
        #   3 Arrays are defined globally for the sparse positions temporaries
        # and two additional bock-sized Arrays are defined locally
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        extra_arrays = 2
        assert len(arrays) == 4 + extra_arrays
        assert all(i._mem_heap and not i._mem_external for i in arrays)
        bns, pbs = assert_blocking(op, {'x0_blk0'})

        arrays = [i for i in FindSymbols().visit(bns['x0_blk0']) if i.is_Array]
        assert len(arrays) == 6
        assert all(not i._mem_external for i in arrays)
        assert len([i for i in arrays if i._mem_heap]) == 6
        vexpanded = 2 if 'openmp' in configuration['language'] else 0
        assert len(FindNodes(VExpanded).visit(pbs['x0_blk0'])) == vexpanded

    @switchconfig(profiling='advanced')
    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_fullopt_w_mpi(self, mode):
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
        (8, 154), (16, 272)
    ])
    def test_opcounts(self, space_order, expected):
        op = self.tti_operator(opt='advanced', space_order=space_order)
        sections = list(op.op_fwd()._profiler._sections.values())
        assert sections[1].sops == expected

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,exp_ops,exp_arrays', [
        (4, 122, 6), (8, 225, 7)
    ])
    def test_opcounts_adjoint(self, space_order, exp_ops, exp_arrays):
        wavesolver = self.tti_operator(space_order=space_order,
                                       opt=('advanced', {'openmp': False}))
        op = wavesolver.op_adj()

        assert op._profiler._sections['section1'].sops == exp_ops
        assert len([i for i in FindSymbols().visit(op) if i.is_Array]) == exp_arrays


class TestTTIv2:

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,expected', [
        (4, 200), (12, 392)
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
        op = Operator(eqns, opt=('advanced', {'openmp': True}))

        # Check code generation
        _, pbs = assert_blocking(op, {'x0_blk0'})
        arrays = FindNodes(VExpanded).visit(pbs['x0_blk0'])
        assert len(arrays) == 4
        assert all(len(i.pointee.shape) == 2 for i in arrays)  # Expected 2D arrays
        sections = list(op._profiler._sections.values())
        assert len(sections) == 2
        assert sections[0].sops == 4
        assert sections[1].sops == expected
