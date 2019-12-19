from sympy import Add, cos, sin, sqrt  # noqa
import numpy as np
import pytest
from unittest.mock import patch
from cached_property import cached_property

from conftest import skipif, EVAL  # noqa
from devito import (Eq, Inc, Constant, Function, TimeFunction, SparseTimeFunction,  # noqa
                    Dimension, SubDimension, Grid, Operator, switchconfig, configuration)
from devito.ir import Stencil, FindSymbols, retrieve_iteration_tree  # noqa
from devito.passes.clusters import common_subexprs_elimination, make_is_time_invariant
from devito.passes.clusters.aliases import collect
from devito.passes.iet import BlockDimension
from devito.symbolics import yreplace, estimate_cost, pow_to_mul
from devito.tools import generator
from devito.types import Scalar, Array

from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, AcquisitionGeometry
from examples.seismic.tti import AnisotropicWaveSolver

pytestmark = skipif(['yask', 'ops'], whole_module=True)


def test_scheduling_after_rewrite():
    """Tests loop scheduling after DSE-induced expression hoisting."""
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
    # simple
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(tu, ti0 + ti1 + 5.)'],
     ['ti0[x, y, z] + ti1[x, y, z] + 5.0']),
    # more ops
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(t0, 0.2)', 'Eq(t1, t0 + 2.)',
      'Eq(tw, 2. + ti0*t1)', 'Eq(tu, (ti0*ti1*t0) + (ti1*tv) + (t1 + ti1)*tw)'],
     ['t0 + 2.0', 't1*ti0[x, y, z] + 2.0', 't1 + ti1[x, y, z]',
      't0*ti0[x, y, z]*ti1[x, y, z]']),
    # wrapped
    (['Eq(ti1, 4.)', 'Eq(ti0, 3.)', 'Eq(t0, 0.2)', 'Eq(t1, t0 + 2.)', 'Eq(tv, 2.4)',
      'Eq(tu, ((ti0*ti1*t0)*tv + (ti0*ti1*tv)*t1))'],
     ['t0 + 2.0', 't0*ti0[x, y, z]*ti1[x, y, z]', 't1*ti0[x, y, z]*ti1[x, y, z]']),
])
def test_yreplace_time_invariants(exprs, expected):
    grid = Grid((3, 3, 3))
    dims = grid.dimensions
    tu = TimeFunction(name="tu", grid=grid, space_order=4).indexify()
    tv = TimeFunction(name="tv", grid=grid, space_order=4).indexify()
    tw = TimeFunction(name="tw", grid=grid, space_order=4).indexify()
    ti0 = Array(name='ti0', shape=(3, 5, 7), dimensions=dims).indexify()
    ti1 = Array(name='ti1', shape=(3, 5, 7), dimensions=dims).indexify()
    t0 = Scalar(name='t0').indexify()
    t1 = Scalar(name='t1').indexify()
    exprs = EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1)
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed, found = yreplace(exprs, make,
                                make_is_time_invariant(exprs),
                                lambda i: estimate_cost(i) > 0)
    assert len(found) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(found, expected))


@pytest.mark.parametrize('exprs,expected', [
    # simple
    (['Eq(tu, (tv + tw + 5.)*(ti0 + ti1) + (t0 + t1)*(ti0 + ti1))'],
     ['ti0[x, y, z] + ti1[x, y, z]',
      'r0*(t0 + t1) + r0*(tv[t, x, y, z] + tw[t, x, y, z] + 5.0)']),
    # across expressions
    (['Eq(tu, tv*4 + tw*5 + tw*5*t0)', 'Eq(tv, tw*5)'],
     ['5*tw[t, x, y, z]', 'r0 + 5*t0*tw[t, x, y, z] + 4*tv[t, x, y, z]', 'r0']),
    # intersecting
    pytest.param(['Eq(tu, ti0*ti1 + ti0*ti1*t0 + ti0*ti1*t0*t1)'],
                 ['ti0*ti1', 'r0', 'r0*t0', 'r0*t0*t1'],
                 marks=pytest.mark.xfail),
    # divisions (== powers with negative exponenet) are always captured
    (['Eq(tu, tv**-1*(tw*5 + tw*5*t0))', 'Eq(ti0, tv**-1*t0)'],
     ['1/tv[t, x, y, z]', 'r0*(5*t0*tw[t, x, y, z] + 5*tw[t, x, y, z])', 'r0*t0']),
    # `compact_temporaries` must detect chains of isolated temporaries
    (['Eq(t0, tv)', 'Eq(t1, t0)', 'Eq(t2, t1)', 'Eq(tu, t2)'],
     ['tv[t, x, y, z]'])
])
def test_common_subexprs_elimination(exprs, expected):
    grid = Grid((3, 3, 3))
    dims = grid.dimensions
    tu = TimeFunction(name="tu", grid=grid, space_order=4).indexify()
    tv = TimeFunction(name="tv", grid=grid, space_order=4).indexify()
    tw = TimeFunction(name="tw", grid=grid, space_order=4).indexify()
    ti0 = Array(name='ti0', shape=(3, 5, 7), dimensions=dims).indexify()
    ti1 = Array(name='ti1', shape=(3, 5, 7), dimensions=dims).indexify()
    t0 = Scalar(name='t0').indexify()
    t1 = Scalar(name='t1').indexify()
    t2 = Scalar(name='t2').indexify()
    counter = generator()
    make = lambda: Scalar(name='r%d' % counter()).indexify()
    processed = common_subexprs_elimination(EVAL(exprs, tu, tv, tw, ti0, ti1, t0, t1, t2),
                                            make)
    assert len(processed) == len(expected)
    assert all(str(i.rhs) == j for i, j in zip(processed, expected))


@pytest.mark.parametrize('expr,expected', [
    ('2*fa[x] + fb[x]', '2*fa[x] + fb[x]'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('fa[x]**2 + fb[x]**3', 'fa[x]*fa[x] + fb[x]*fb[x]*fb[x]'),
    ('3*fa[x]**4', '3*(fa[x]*fa[x]*fa[x]*fa[x])'),
    ('fa[x]**2', 'fa[x]*fa[x]'),
    ('1/(fa[x]**2)', '1/(fa[x]*fa[x])'),
    ('1/(fa[x] + fb[x])', '1/(fa[x] + fb[x])'),
    ('3*sin(fa[x])**2', '3*(sin(fa[x])*sin(fa[x]))'),
])
def test_pow_to_mul(expr, expected):
    grid = Grid((4, 5))
    x, y = grid.dimensions
    fa = Function(name='fa', grid=grid, dimensions=(x,), shape=(4,))  # noqa
    fb = Function(name='fb', grid=grid, dimensions=(x,), shape=(4,))  # noqa
    assert str(pow_to_mul(eval(expr))) == expected


@pytest.mark.parametrize('exprs,expected', [
    # none (different distance)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x])'],
     {'fa[x] + fb[x]': 'None', 'fa[x+1] + fb[x]': 'None'}),
    # none (different dimension)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] + fb[y])'],
     {'fa[x] + fb[x]': 'None', 'fa[x] + fb[y]': 'None'}),
    # none (different operation)
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x] - fb[x])'],
     {'fa[x] + fb[x]': 'None', 'fa[x] - fb[x]': 'None'}),
    # simple
    (['Eq(t0, fa[x] + fb[x])', 'Eq(t1, fa[x+1] + fb[x+1])', 'Eq(t2, fa[x-1] + fb[x-1])'],
     {'fa[x] + fb[x]': 'Stencil([(x, {-1, 0, 1})])'}),
    # 2D simple
    (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(t1, fc[x+1,y+1] + fd[x+1,y+1])'],
     {'fc[x,y] + fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with stride
    (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])'],
     {'fc[x,y] + fd[x+1,y+2]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with subdimensions
    (['Eq(t0, fc[xi,yi] + fd[xi+1,yi+2])', 'Eq(t1, fc[xi+1,yi+1] + fd[xi+2,yi+3])'],
     {'fc[xi,yi] + fd[xi+1,yi+2]': 'Stencil([(xi, {0, 1}), (yi, {0, 1})])'}),
    # 2D with constant access
    (['Eq(t0, fc[x,y]*fc[x,0] + fd[x,y])', 'Eq(t1, fc[x+1,y+1]*fc[x+1,0] + fd[x+1,y+1])'],
     {'fc[x,y]*fc[x,0] + fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with multiple, non-zero, constant accesses
    (['Eq(t0, fc[x,y]*fc[x,0] + fd[x,y]*fc[x,1])',
      'Eq(t1, fc[x+1,y+1]*fc[x+1,0] + fd[x+1,y+1]*fc[x+1,1])'],
     {'fc[x,0]*fc[x,y] + fc[x,1]*fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # 2D with different shapes
    (['Eq(t0, fc[x,y]*fa[x] + fd[x,y])', 'Eq(t1, fc[x+1,y+1]*fa[x+1] + fd[x+1,y+1])'],
     {'fc[x,y]*fa[x] + fd[x,y]': 'Stencil([(x, {0, 1}), (y, {0, 1})])'}),
    # complex (two 2D aliases with stride inducing relaxation)
    (['Eq(t0, fc[x,y] + fd[x+1,y+2])', 'Eq(t1, fc[x+1,y+1] + fd[x+2,y+3])',
      'Eq(t2, fc[x-2,y-2]*3. + fd[x+2,y+2])', 'Eq(t3, fc[x-4,y-4]*3. + fd[x,y])'],
     {'fc[x,y] + fd[x+1,y+2]': 'Stencil([(x, {-1, 0, 1}), (y, {-1, 0, 1})])',
      '3.*fc[x-3,y-3] + fd[x+1,y+1]': 'Stencil([(x, {-1, 0, 1}), (y, {-1, 0, 1})])'}),
])
def test_collect_aliases(exprs, expected):
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions  # noqa
    xi, yi = grid.interior.dimensions  # noqa

    t0 = Scalar(name='t0')  # noqa
    t1 = Scalar(name='t1')  # noqa
    t2 = Scalar(name='t2')  # noqa
    t3 = Scalar(name='t3')  # noqa
    fa = Function(name='fa', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fb = Function(name='fb', grid=grid, shape=(4,), dimensions=(x,))  # noqa
    fc = Function(name='fc', grid=grid)  # noqa
    fd = Function(name='fd', grid=grid)  # noqa

    # List/dict comprehension would need explicit locals/globals mappings to eval
    for i, e in enumerate(list(exprs)):
        exprs[i] = eval(e)
    for k, v in list(expected.items()):
        expected[eval(k)] = eval(v)

    aliases = collect(exprs)

    assert len(aliases) > 0

    for k, v in aliases.items():
        assert ((len(v.aliased) == 1 and expected[k] is None) or
                v.anti_stencil == expected[k])


@pytest.mark.parametrize('expr,expected,estimate', [
    ('Eq(t0, t1)', 0, False),
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
    ('Eq(t0, cos(t1*t2))', 51, True),
    ('Eq(t0, t1**3)', 2, True),
    ('Eq(t0, t1**4)', 3, True),
    ('Eq(t0, t2*t1**-1)', 26, True),
    ('Eq(t0, t1**t2)', 50, True),
])
def test_estimate_cost(expr, expected, estimate):
    # Note: integer arithmetic isn't counted
    grid = Grid(shape=(4, 4))
    x, y = grid.dimensions  # noqa

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


@pytest.mark.parametrize('dse', ['noop', 'basic', 'advanced', 'aggressive'])
@pytest.mark.parametrize('dle', ['noop', 'advanced'])
def test_time_dependent_split(dse, dle):
    grid = Grid(shape=(10, 10))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2, save=3)
    v = TimeFunction(name='v', grid=grid, time_order=2, space_order=0, save=3)

    # The second equation needs a full loop over x/y for u then
    # a full one over x.y for v
    eq = [Eq(u.forward, 2 + grid.time_dim),
          Eq(v.forward, u.forward.dx + u.forward.dy + 1)]
    op = Operator(eq, dse=dse, dle=dle)

    trees = retrieve_iteration_tree(op)
    assert len(trees) == 2

    op()

    assert np.allclose(u.data[2, :, :], 3.0)
    assert np.allclose(v.data[1, 1:-1, 1:-1], 1.0)


class TestAliases(object):

    @patch("devito.passes.clusters.aliases.MIN_COST_ALIAS", 1)
    def test_full_shape_after_blocking(self):
        """
        Check the shape of the Array used to store a DSE-captured aliasing
        expression. The shape is impacted by loop blocking, which reduces the
        required write-to space.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions  # noqa
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        # Leads to 3D aliases
        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3*f + 1))
        op0 = Operator(eqn, dse='noop', dle=('advanced', {'openmp': True}))
        op1 = Operator(eqn, dse='aggressive', dle=('advanced', {'openmp': True}))

        x0_blk_size = op1.parameters[-3]
        y0_blk_size = op1.parameters[-2]
        z_size = op1.parameters[4]

        # Check Array shape
        arrays = [i for i in FindSymbols().visit(op1._func_table['bf0'].root)
                  if i.is_Array]
        assert len(arrays) == 1
        a = arrays[0]
        assert len(a.dimensions) == 3
        assert a.halo == ((1, 1), (1, 1), (1, 1))
        assert Add(*a.symbolic_shape[0].args) == x0_blk_size + 2
        assert Add(*a.symbolic_shape[1].args) == y0_blk_size + 2
        assert Add(*a.symbolic_shape[2].args) == z_size + 2
        # Check numerical output
        op0(time_M=1)
        exp = np.copy(u.data[:])
        u.data_with_halo[:] = 0.
        op1(time_M=1)
        assert np.all(u.data == exp)

    @patch("devito.passes.clusters.aliases.MIN_COST_ALIAS", 1)
    def test_contracted_shape_after_blocking(self):
        """
        Like `test_full_alias_shape_after_blocking`, but a different
        Operator is used, leading to contracted Arrays (2D instead of 3D).
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions  # noqa
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        # Leads to 2D aliases
        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x, y+1, z+1])*3*f +
                             (u[t, x, y+2, z+2] + u[t, x, y+3, z+3])*3*f + 1))
        op0 = Operator(eqn, dse='noop', dle=('advanced', {'openmp': True}))
        op1 = Operator(eqn, dse='aggressive', dle=('advanced', {'openmp': True}))

        y0_blk_size = op1.parameters[-2]
        z_size = op1.parameters[3]

        arrays = [i for i in FindSymbols().visit(op1._func_table['bf0'].root)
                  if i.is_Array]
        assert len(arrays) == 1
        a = arrays[0]
        assert len(a.dimensions) == 2
        assert a.halo == ((1, 1), (1, 1))
        assert Add(*a.symbolic_shape[0].args) == y0_blk_size + 2
        assert Add(*a.symbolic_shape[1].args) == z_size + 2
        # Check numerical output
        op0(time_M=1)
        exp = np.copy(u.data[:])
        u.data_with_halo[:] = 0.
        op1(time_M=1)
        assert np.all(u.data == exp)

    @patch("devito.passes.clusters.aliases.MIN_COST_ALIAS", 1)
    def test_full_shape_with_subdims(self):
        """
        Like `test_full_alias_shape_after_blocking`, but SubDomains (and therefore
        SubDimensions) are used. Nevertheless, the temporary shape should still be
        dictated by the root Dimensions.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions  # noqa
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        # Leads to 3D aliases
        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3*f + 1),
                 subdomain=grid.interior)
        op0 = Operator(eqn, dse='noop', dle=('advanced', {'openmp': True}))
        op1 = Operator(eqn, dse='aggressive', dle=('advanced', {'openmp': True}))

        xi0_blk_size = op1.parameters[-3]
        yi0_blk_size = op1.parameters[-2]
        z_size = op1.parameters[4]

        # Check Array shape
        arrays = [i for i in FindSymbols().visit(op1._func_table['bf0'].root)
                  if i.is_Array]
        assert len(arrays) == 1
        a = arrays[0]
        assert len(a.dimensions) == 3
        assert a.halo == ((1, 1), (1, 1), (1, 1))
        assert Add(*a.symbolic_shape[0].args) == xi0_blk_size + 2
        assert Add(*a.symbolic_shape[1].args) == yi0_blk_size + 2
        assert Add(*a.symbolic_shape[2].args) == z_size + 2
        # Check numerical output
        op0(time_M=1)
        exp = np.copy(u.data[:])
        u.data_with_halo[:] = 0.
        op1(time_M=1)
        assert np.all(u.data == exp)

    def test_composite(self):
        """
        Check that composite alias are optimized away through "smaller" aliases.

        Examples
        --------
        Instead of the following:

            t0 = a[x, y]
            t1 = b[x, y]
            t2 = a[x+1, y+1]*b[x, y]
            out = t0 + t1 + t2  # pseudocode

        We should get:

            t0 = a[x, y]
            t1 = b[x, y]
            out = t0 + t1 + t0[x+1,y+1]*t1[x, y]  # pseudocode
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions  # noqa

        u = TimeFunction(name='u', grid=grid)
        u.data[:] = 1.
        g = Function(name='g', grid=grid)
        g.data[:] = 2.

        expr = (cos(g)*cos(g) +
                sin(g)*sin(g) +
                sin(g)*cos(g) +
                sin(g[x + 1, y + 1])*cos(g[x + 1, y + 1]))*u

        op0 = Operator(Eq(u.forward, expr), dse='noop')
        op1 = Operator(Eq(u.forward, expr), dse='aggressive')

        # We expect two temporary Arrays, one for `cos(g)` and one for `sin(g)`
        arrays = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)
        # Check numerical output
        op0(time_M=1)
        exp = np.copy(u.data[:])
        u.data[:] = 1.
        op1(time_M=1)
        assert np.allclose(u.data, exp.data, rtol=10e-7)

    @pytest.mark.xfail(reason="Cannot deal with nested aliases yet")
    def test_nested(self):
        """
        Check that nested aliases are optimized away through "smaller" aliases.

        Examples
        --------
        Given the expression

            sqrt(cos(a[x, y]))

        We should get

            t0 = cos(a[x,y])
            t1 = sqrt(t0)
            out = t1  # pseudocode
        """
        grid = Grid(shape=(3, 3))
        x, y = grid.dimensions  # noqa

        u = TimeFunction(name='u', grid=grid)
        g = Function(name='g', grid=grid)

        op = Operator(Eq(u.forward, u + sin(cos(g)) + sin(cos(g[x+1, y+1]))))

        # We expect two temporary Arrays: `r1 = cos(g)` and `r2 = sqrt(r1)`
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    @patch("devito.passes.clusters.aliases.MIN_COST_ALIAS", 1)
    def test_from_different_nests(self):
        """
        Check that aliases arising from two sets of equations A and B,
        characterized by a flow dependence, are scheduled within A's and B's
        loop nests respectively.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions  # noqa
        t = grid.stepping_dim
        i = Dimension(name='i')

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        g = Function(name='g', shape=(3,), dimensions=(i,))
        g.data[:] = 2.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        v = TimeFunction(name='v', grid=grid, space_order=3)

        # Leads to 3D aliases
        eqns = [Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3*f +
                               (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3*f + 1)),
                Inc(u[t+1, i, i, i], g + 1),
                Eq(v.forward, ((v[t, x, y, z] + v[t, x+1, y+1, z+1])*3*u.forward +
                               (v[t, x+2, y+2, z+2] + v[t, x+3, y+3, z+3])*3*u.forward +
                               1))]
        op0 = Operator(eqns, dse='noop', dle=('noop', {'openmp': True}))
        op1 = Operator(eqns, dse='aggressive', dle=('advanced', {'openmp': True}))

        # Check code generation
        assert 'bf0' in op1._func_table
        assert 'bf1' in op1._func_table
        trees = retrieve_iteration_tree(op1._func_table['bf0'].root)
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].body[0].write.is_Array
        assert trees[1][-1].nodes[0].body[0].write is u
        trees = retrieve_iteration_tree(op1._func_table['bf1'].root)
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].body[0].write.is_Array
        assert trees[1][-1].nodes[0].body[0].write is v

        # Check numerical output
        op0(time_M=1)
        exp = np.copy(u.data[:])
        u.data_with_halo[:] = 0.
        op1(time_M=1)
        assert np.all(u.data == exp)

    @switchconfig(autopadding=True, platform='knl7210')  # Platform is to fix pad value
    @patch("devito.passes.clusters.aliases.MIN_COST_ALIAS", 1)
    def test_minimize_remainders_due_to_autopadding(self):
        """
        Check that the bounds of the Iteration computing the DSE-captured aliasing
        expressions are relaxed (i.e., slightly larger) so that backend-compiler-generated
        remainder loops are avoided.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions  # noqa
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        f.data_with_halo[:] = 1.
        u = TimeFunction(name='u', grid=grid, space_order=3)
        u.data_with_halo[:] = 0.

        # Leads to 3D aliases
        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3*f + 1))
        op0 = Operator(eqn, dse='noop', dle=('advanced', {'openmp': False}))
        op1 = Operator(eqn, dse='aggressive', dle=('advanced', {'openmp': False}))

        x0_blk_size = op1.parameters[-2]
        y0_blk_size = op1.parameters[-1]
        z_size = op1.parameters[4]

        # Check Array shape
        arrays = [i for i in FindSymbols().visit(op1._func_table['bf0'].root)
                  if i.is_Array]
        assert len(arrays) == 1
        a = arrays[0]
        assert len(a.dimensions) == 3
        assert a.halo == ((1, 1), (1, 1), (1, 1))
        assert a.padding == ((0, 0), (0, 0), (0, 30))
        assert Add(*a.symbolic_shape[0].args) == x0_blk_size + 2
        assert Add(*a.symbolic_shape[1].args) == y0_blk_size + 2
        assert Add(*a.symbolic_shape[2].args) == z_size + 32

        # Check loop bounds
        trees = retrieve_iteration_tree(op1._func_table['bf0'].root)
        assert len(trees) == 2
        expected_rounded = trees[0].inner
        assert expected_rounded.symbolic_max ==\
            z.symbolic_max + (z.symbolic_max - z.symbolic_min + 3) % 16 + 1

        # Check numerical output
        op0(time_M=1)
        exp = np.copy(u.data[:])
        u.data_with_halo[:] = 0.
        op1(time_M=1)
        assert np.all(u.data == exp)

    def test_catch_largest_time_invariant(self):
        """
        Make sure the DSE extracts the largest time-invariant sub-expressions
        such that its operation count exceeds a certain threshold.
        """
        grid = Grid((10, 10))

        a = Function(name="a", grid=grid, space_order=4)
        b = Function(name="b", grid=grid, space_order=4)
        c = Function(name="c", grid=grid, space_order=4)
        d = Function(name="d", grid=grid, space_order=4)

        e = TimeFunction(name="e", grid=grid, space_order=4)

        deriv = (sqrt((a - 2*b)/c) * e.dx).dy + (sqrt((d - 2*c)/a) * e.dy).dx

        op = Operator(Eq(e.forward, deriv + e))

        # We expect two temporary Arrays, one for each `sqrt` subexpr
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)

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

        op = Operator(eqns)

        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 3
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_hoisting_if_coupled(self):
        """
        Test that coupled aliases are successfully hoisted out of the time loop.
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
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)

    def test_drop_redundants_after_fusion(self):
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

        op = Operator(eqns)

        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 2
        assert all(i._mem_heap and not i._mem_external for i in arrays)


# Acoustic

def run_acoustic_forward(dse=None):
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

    solver = AcousticWaveSolver(model, geometry, dse=dse, dle='noop')
    rec, u, _ = solver.forward(save=False)

    return u, rec


def test_acoustic_rewrite_basic():
    ret1 = run_acoustic_forward(dse=None)
    ret2 = run_acoustic_forward(dse='basic')

    assert np.allclose(ret1[0].data, ret2[0].data, atol=10e-5)
    assert np.allclose(ret1[1].data, ret2[1].data, atol=10e-5)


def test_custom_rewriter():
    ret1 = run_acoustic_forward(dse=None)
    ret2 = run_acoustic_forward(dse=('extract_sop', 'factorize',
                                     'extract_invariants', 'gcse'))

    assert np.allclose(ret1[0].data, ret2[0].data, atol=10e-5)
    assert np.allclose(ret1[1].data, ret2[1].data, atol=10e-5)


# TTI
class TestTTI(object):

    # TTI layered model for the tti test, no need for a smooth interace bewtween
    # the two layer as the dse/compiler is tested not the physical prettiness
    # of the result, saves testing time
    _model = demo_model('layers-tti', nlayers=3, nbl=10, space_order=4,
                        shape=(50, 50, 50), spacing=(20., 20., 20.),
                        smooth=False)

    @cached_property
    def model(self):
        return self._model

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

    def tti_operator(self, dse=False, space_order=4):
        return AnisotropicWaveSolver(self.model, self.geometry,
                                     space_order=space_order, dse=dse)

    @cached_property
    def tti_nodse(self):
        operator = self.tti_operator(dse=None)
        rec, u, v, _ = operator.forward()
        return v, rec

    def test_tti_rewrite_basic(self):
        operator = self.tti_operator(dse='basic')
        rec, u, v, _ = operator.forward()

        assert np.allclose(self.tti_nodse[0].data, v.data, atol=10e-3)
        assert np.allclose(self.tti_nodse[1].data, rec.data, atol=10e-3)

    def test_tti_rewrite_advanced(self):
        operator = self.tti_operator(dse='advanced')
        rec, u, v, _ = operator.forward()

        assert np.allclose(self.tti_nodse[0].data, v.data, atol=10e-1)
        assert np.allclose(self.tti_nodse[1].data, rec.data, atol=10e-1)

    def test_tti_rewrite_aggressive(self):
        operator = self.tti_operator(dse='aggressive')
        rec, u, v, _ = operator.forward(kernel='centered')

        assert np.allclose(self.tti_nodse[0].data, v.data, atol=10e-1)
        assert np.allclose(self.tti_nodse[1].data, rec.data, atol=10e-1)

        # Also check that DLE's loop blocking with DSE=aggressive does the right thing
        # There should be exactly two BlockDimensions; bugs in the past were generating
        # either code with no blocking (zero BlockDimensions) or code with four
        # BlockDimensions (i.e., Iteration folding was somewhat broken)
        op = operator.op_fwd(kernel='centered')
        block_dims = [i for i in op.dimensions if isinstance(i, BlockDimension)]
        assert len(block_dims) == 2

        # Also, in this operator, we expect six temporary Arrays:
        # * four Arrays are allocated on the heap
        # * two Arrays are allocated on the stack and only appear within an efunc
        arrays = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(arrays) == 5
        assert all(i._mem_heap and not i._mem_external for i in arrays)
        arrays = [i for i in FindSymbols().visit(op._func_table['bf0'].root)
                  if i.is_Array]
        assert len(arrays) == 7
        assert all(not i._mem_external for i in arrays)
        assert len([i for i in arrays if i._mem_heap]) == 5
        assert len([i for i in arrays if i._mem_stack]) == 2

    @skipif(['nompi'])
    @pytest.mark.parallel(mode=[(1, 'full')])
    def test_tti_rewrite_aggressive_wmpi(self):
        tti_nodse = self.tti_operator(dse=None)
        rec0, u0, v0, _ = tti_nodse.forward(kernel='centered')
        tti_agg = self.tti_operator(dse='aggressive')
        rec1, u1, v1, _ = tti_agg.forward(kernel='centered')

        assert np.allclose(v0.data, v1.data, atol=10e-1)
        assert np.allclose(rec0.data, rec1.data, atol=10e-1)

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,expected', [
        (8, 174), (16, 306)
    ])
    def test_tti_rewrite_aggressive_opcounts(self, space_order, expected):
        op = self.tti_operator(dse='aggressive', space_order=space_order)
        sections = list(op.op_fwd(kernel='centered')._profiler._sections.values())
        assert sections[1].sops == expected

    @switchconfig(profiling='advanced')
    @pytest.mark.parametrize('space_order,expected', [
        (4, 194), (12, 386)
    ])
    def test_tti_v2_rewrite_aggressive_opcounts(self, space_order, expected):
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
        op = Operator(eqns, dse='aggressive')

        sections = list(op._profiler._sections.values())
        assert len(sections) == 2
        assert sections[0].sops == 4
        assert sections[1].sops == expected
