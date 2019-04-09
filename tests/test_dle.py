from functools import reduce
from operator import mul

import numpy as np
import pytest

from conftest import EVAL, skipif
from devito import (Grid, Function, TimeFunction, SparseTimeFunction, SubDimension,
                    Eq, Operator, solve)
from devito.dle import BlockDimension, NThreads, transform
from devito.dle.parallelizer import nhyperthreads
from devito.exceptions import InvalidArgument
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Expression, Iteration, Conditional, FindNodes,
                           iet_analyze, retrieve_iteration_tree)
from devito.tools import as_tuple
from unittest.mock import patch

pytestmark = skipif(['yask', 'ops'])


def get_blocksizes(op, dle, grid, blockshape, level=0):
    blocksizes = {'%s0_blk%d_size' % (d, level): v
                  for d, v in zip(grid.dimensions, blockshape)}
    blocksizes = {k: v for k, v in blocksizes.items() if k in op._known_arguments}
    # Sanity check
    if grid.dim == 1 or len(blockshape) == 0:
        assert len(blocksizes) == 0
        return {}
    try:
        if dle[1].get('blockinner'):
            assert len(blocksizes) >= 1
            if grid.dim == len(blockshape):
                assert len(blocksizes) == len(blockshape)
            else:
                assert len(blocksizes) <= len(blockshape)
        return blocksizes
    except AttributeError:
        assert len(blocksizes) == 0
        return {}


def _new_operator1(shape, blockshape=None, dle=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape, dtype=np.int32)
    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = Function(name='outfield', grid=grid)

    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=dle)

    blocksizes = get_blocksizes(op, dle, grid, blockshape)
    op(infield=infield, outfield=outfield, **blocksizes)

    return outfield, op


def _new_operator2(shape, time_order, blockshape=None, dle=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape, dtype=np.int32)
    infield = TimeFunction(name='infield', grid=grid, time_order=time_order)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)
    outfield = TimeFunction(name='outfield', grid=grid, time_order=time_order)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)
    op = Operator(stencil, dle=dle)

    blocksizes = get_blocksizes(op, dle, grid, blockshape)
    op(infield=infield, outfield=outfield, t=10, **blocksizes)

    return outfield, op


def _new_operator3(shape, blockshape0=None, blockshape1=None, dle=None):
    blockshape0 = as_tuple(blockshape0)
    blockshape1 = as_tuple(blockshape1)

    grid = Grid(shape=shape, dtype=np.float64)
    spacing = 0.1
    a = 0.5
    c = 0.5
    dx2, dy2 = spacing**2, spacing**2
    dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

    # Allocate the grid and set initial condition
    # Note: This should be made simpler through the use of defaults
    u = TimeFunction(name='u', grid=grid, time_order=1, space_order=(2, 2, 2))
    u.data[0, :] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)

    # Derive the stencil according to devito conventions
    eqn = Eq(u.dt, a * (u.dx2 + u.dy2) - c * (u.dxl + u.dyl))
    stencil = solve(eqn, u.forward)
    op = Operator(Eq(u.forward, stencil), dle=dle)

    blocksizes0 = get_blocksizes(op, dle, grid, blockshape0, 0)
    blocksizes1 = get_blocksizes(op, dle, grid, blockshape1, 1)
    op.apply(u=u, t=10, dt=dt, **blocksizes0, **blocksizes1)

    return u.data[1, :], op


@pytest.mark.parametrize("blockinner,exp_calls,exp_iters", [
    (False, 4, 5),
    (True, 8, 6)
])
@patch("devito.dle.parallelizer.Ompizer.COLLAPSE_NCORES", 1)
def test_cache_blocking_structure(blockinner, exp_calls, exp_iters):
    # Check code structure
    _, op = _new_operator1((10, 31, 45), dle=('blocking', {'blockalways': True,
                                                           'blockinner': blockinner}))
    calls = FindNodes(Call).visit(op)
    assert len(calls) == exp_calls
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree) == exp_iters
    assert isinstance(tree[0].dim, BlockDimension)
    assert isinstance(tree[1].dim, BlockDimension)
    if blockinner:
        assert isinstance(tree[2].dim, BlockDimension)
    else:
        assert not isinstance(tree[2].dim, BlockDimension)
    assert not isinstance(tree[3].dim, BlockDimension)
    assert not isinstance(tree[4].dim, BlockDimension)

    # Check presence of openmp pragmas at the right place
    _, op = _new_operator1((10, 31, 45), dle=('blocking',
                                              {'openmp': True,
                                               'blockalways': True,
                                               'blockinner': blockinner}))
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree.root.pragmas) == 1
    assert 'omp for' in tree.root.pragmas[0].value
    # Also, with omp parallelism enabled, the step increment must be != 0
    # to avoid omp segfaults at scheduling time (only certain omp implementations,
    # including Intel's)
    conditionals = FindNodes(Conditional).visit(op._func_table['bf0'].root)
    assert len(conditionals) == 1
    conds = conditionals[0].condition.args
    expected_guarded = tree[:2+blockinner]
    assert len(conds) == len(expected_guarded)
    assert all(i.lhs == j.step for i, j in zip(conds, expected_guarded))


def test_cache_blocking_structure_subdims():
    """
    Test that:

        * With local SubDimensions no-blocking is expected.
        * With non-local SubDimensions, blocking is expected.
    """
    grid = Grid(shape=(4, 4, 4))
    x, y, z = grid.dimensions
    t = grid.stepping_dim
    xl = SubDimension.left(name='xl', parent=x, thickness=4)

    f = TimeFunction(name='f', grid=grid)

    assert xl.local

    # Local SubDimension -> no blocking expected
    op = Operator(Eq(f[t+1, xl, y, z], f[t, xl, y, z] + 1))
    assert len(op._func_table) == 0

    # Non-local SubDimension -> blocking expected
    op = Operator(Eq(f.forward, f + 1, subdomain=grid.interior))
    trees = retrieve_iteration_tree(op._func_table['bf0'].root)
    assert len(trees) == 1
    tree = trees[0]
    assert len(tree) == 5
    assert isinstance(tree[0].dim, BlockDimension) and tree[0].dim.root is x
    assert isinstance(tree[1].dim, BlockDimension) and tree[1].dim.root is y
    assert not isinstance(tree[2].dim, BlockDimension)


@pytest.mark.parametrize("shape", [(10,), (10, 45), (10, 31, 45)])
@pytest.mark.parametrize("blockshape", [(2,), (7,), (3, 3), (2, 9, 1)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_no_time_loop(shape, blockshape, blockinner):
    wo_blocking, _ = _new_operator1(shape, dle='noop')
    w_blocking, _ = _new_operator1(shape, blockshape, dle=('blocking',
                                                           {'blockalways': True,
                                                            'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape", [(20, 33), (45, 31, 45)])
@pytest.mark.parametrize("time_order", [2])
@pytest.mark.parametrize("blockshape", [2, (13, 20), (11, 15, 23)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_time_loop(shape, time_order, blockshape, blockinner):
    wo_blocking, _ = _new_operator2(shape, time_order, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   dle=('blocking', {'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((25, 25, 46), (25, 25, 46)),
    ((25, 25, 46), (7, 25, 46)),
    ((25, 25, 46), (25, 25, 7)),
    ((25, 25, 46), (25, 7, 46)),
    ((25, 25, 46), (5, 25, 7)),
    ((25, 25, 46), (10, 3, 46)),
    ((25, 25, 46), (25, 7, 11)),
    ((25, 25, 46), (8, 2, 4)),
    ((25, 25, 46), (2, 4, 8)),
    ((25, 25, 46), (4, 8, 2)),
    ((25, 46), (25, 7)),
    ((25, 46), (7, 46))
])
def test_cache_blocking_edge_cases(shape, blockshape):
    time_order = 2
    wo_blocking, _ = _new_operator2(shape, time_order, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order, blockshape,
                                   dle=('blocking', {'blockinner': True}))
    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((3, 3), (3, 3)),
    ((4, 4), (3, 4)),
    ((5, 5), (3, 4)),
    ((6, 6), (3, 4)),
    ((7, 7), (3, 4)),
    ((8, 8), (3, 4)),
    ((9, 9), (3, 4)),
    ((10, 10), (3, 4)),
    ((11, 11), (3, 4)),
    ((12, 12), (3, 4)),
    ((13, 13), (3, 4)),
    ((14, 14), (3, 4)),
    ((15, 15), (3, 4))
])
def test_cache_blocking_edge_cases_highorder(shape, blockshape):
    wo_blocking, a = _new_operator3(shape, dle='noop')
    w_blocking, b = _new_operator3(shape, blockshape, dle=('blocking',
                                                           {'blockinner': True}))

    assert np.allclose(wo_blocking, w_blocking, rtol=1e-12)


@pytest.mark.parametrize("blockshape0,blockshape1,exception", [
    ((24, 24, 40), (24, 24, 40), False),
    ((24, 24, 40), (4, 4, 4), False),
    ((24, 24, 40), (8, 8, 8), False),
    ((20, 20, 12), (4, 4, 4), False),
    ((28, 32, 16), (14, 16, 8), False),
    ((12, 12, 60), (4, 12, 4), False),
    ((12, 12, 60), (4, 5, 4), True),  # not a perfect divisor
    ((12, 12, 60), (24, 4, 4), True),  # bigger than outer block
])
def test_cache_blocking_hierarchical(blockshape0, blockshape1, exception):
    shape = (51, 102, 71)

    wo_blocking, a = _new_operator3(shape, dle='noop')
    try:
        w_blocking, b = _new_operator3(shape, blockshape0, blockshape1,
                                       dle=('blocking', {'blockinner': True,
                                                         'blocklevels': 2}))
        assert not exception
        assert np.all(wo_blocking == w_blocking)
    except InvalidArgument:
        assert exception
    except:
        assert False


class TestNodeParallelism(object):

    @pytest.mark.parametrize('exprs,expected', [
        # trivial 1D
        (['Eq(fa[x], fa[x] + fb[x])'],
         (True, False)),
        # trivial 1D
        (['Eq(t0, fa[x] + fb[x])', 'Eq(fa[x], t0 + 1)'],
         (True, False)),
        # trivial 2D
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y], t0 + 1)'],
         (True, False)),
        # outermost parallel, innermost sequential
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x,y+1], t0 + 1)'],
         (True, False)),
        # outermost sequential, innermost parallel
        (['Eq(t0, fc[x,y] + fd[x,y])', 'Eq(fc[x+1,y], t0 + 1)'],
         (False, True)),
        # outermost sequential, innermost parallel
        (['Eq(fc[x,y], fc[x+1,y+1] + fc[x-1,y])'],
         (False, True)),
        # outermost parallel w/ repeated dimensions, but the compiler is conservative
        # and makes it sequential, as it doesn't like what happens in the inner dims,
        # where `x`, rather than `y`, is used
        (['Eq(t0, fc[x,x] + fd[x,y+1])', 'Eq(fc[x,x], t0 + 1)'],
         (False, False)),
        # outermost sequential w/ repeated dimensions
        (['Eq(t0, fc[x,x] + fd[x,y+1])', 'Eq(fc[x,x+1], t0 + 1)'],
         (False, False)),
        # outermost sequential, innermost sequential (classic skewing example)
        (['Eq(fc[x,y], fc[x,y+1] + fc[x-1,y])'],
         (False, False)),
        # outermost parallel, innermost sequential w/ double tensor write
        (['Eq(fc[x,y], fc[x,y+1] + fd[x-1,y])', 'Eq(fd[x-1,y+1], fd[x-1,y] + fc[x,y+1])'],
         (True, False)),
        # outermost sequential, innermost parallel w/ mixed dimensions
        (['Eq(fc[x+1,y], fc[x,y+1] + fc[x,y])', 'Eq(fc[x+1,y], 2. + fc[x,y+1])'],
         (False, True)),
    ])
    def test_iterations_ompized(self, fa, fb, fc, fd, t0, t1, t2, t3,
                                exprs, expected, iters):
        scope = [fa, fb, fc, fd, t0, t1, t2, t3]
        node_exprs = [Expression(DummyEq(EVAL(i, *scope))) for i in exprs]
        ast = iters[6](iters[7](node_exprs))

        ast = iet_analyze(ast)

        iet, _ = transform(ast, mode='openmp')
        iterations = FindNodes(Iteration).visit(iet)
        assert len(iterations) == len(expected)

        # Check for presence of pragma omp
        for i, j in zip(iterations, expected):
            pragmas = i.pragmas
            if j is True:
                assert len(pragmas) == 1
                pragma = pragmas[0]
                assert 'omp for' in pragma.value
            else:
                for k in pragmas:
                    assert 'omp for' not in k.value

    def test_dynamic_nthreads(self):
        grid = Grid(shape=(16, 16, 16))
        f = TimeFunction(name='f', grid=grid)

        op = Operator(Eq(f.forward, f + 1.), dle='openmp')

        # Check num_threads appears in the generated code
        # Not very elegant, but it does the trick
        assert 'num_threads(nthreads)' in str(op)

        # Check `op` accepts the `nthreads` kwarg
        op.apply(time=0)
        op.apply(time_m=1, time_M=1, nthreads=4)
        assert np.all(f.data[0] == 2.)

        # Check the actual value assumed by `nthreads`
        assert op.arguments(time=0)['nthreads'] == NThreads.default_value()
        assert op.arguments(time=0, nthreads=123)['nthreads'] == 123  # user supplied

    @pytest.mark.parametrize('eq,expected,blocking', [
        ('Eq(f, 2*f)', [2, 0, 0], False),
        ('Eq(u, 2*u)', [0, 2, 0, 0], False),
        ('Eq(u, 2*u)', [3, 0, 0, 0, 0, 0], True)
    ])
    @patch("devito.dle.parallelizer.Ompizer.COLLAPSE_NCORES", 1)
    @patch("devito.dle.parallelizer.Ompizer.COLLAPSE_WORK", 0)
    def test_collapsing(self, eq, expected, blocking):
        grid = Grid(shape=(3, 3, 3))

        f = Function(name='f', grid=grid)  # noqa
        u = TimeFunction(name='u', grid=grid)  # noqa

        eq = eval(eq)

        if blocking:
            op = Operator(eq, dle=('blocking', 'openmp', {'blockinner': True}))
            iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        else:
            op = Operator(eq, dle='openmp')
            iterations = FindNodes(Iteration).visit(op)

        assert len(iterations) == len(expected)

        # Check for presence of pragma omp + collapse clause
        for i, j in zip(iterations, expected):
            if j > 0:
                assert len(i.pragmas) == 1
                pragma = i.pragmas[0]
                assert 'omp for collapse(%d)' % j in pragma.value
            else:
                for k in i.pragmas:
                    assert 'omp for collapse' not in k.value

    def test_scheduling(self):
        """
        Affine iterations -> #pragma omp ... schedule(static,1) ...
        Non-affine iterations -> #pragma omp ... schedule(static) ...
        """
        grid = Grid(shape=(11, 11))

        u = TimeFunction(name='u', grid=grid, time_order=2, save=5, space_order=0)
        sf1 = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=5)

        eqns = [Eq(u.forward, u + 1)]
        eqns += sf1.interpolate(u)

        op = Operator(eqns, dle='openmp')

        iterations = FindNodes(Iteration).visit(op)
        assert len(iterations) == 4
        assert iterations[1].is_Affine
        assert 'schedule(static,1)' in iterations[1].pragmas[0].value
        assert not iterations[3].is_Affine
        assert 'schedule(static)' in iterations[3].pragmas[0].value


class TestNestedParallelism(object):

    @patch("devito.dle.parallelizer.Ompizer.NESTED", 0)
    @patch("devito.dle.parallelizer.Ompizer.COLLAPSE", 10000)
    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('blocking', 'openmp'))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        assert iterations[0].pragmas[0].value == 'omp for collapse(1) schedule(static,1)'
        assert iterations[2].pragmas[0].value ==\
            ('omp parallel for collapse(1) schedule(static,1) num_threads(%d)'
             % nhyperthreads())

    @patch("devito.dle.parallelizer.Ompizer.NESTED", 0)
    @patch("devito.dle.parallelizer.Ompizer.COLLAPSE_NCORES", 1)
    @patch("devito.dle.parallelizer.Ompizer.COLLAPSE_WORK", 0)
    def test_collapsing(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), dle=('blocking', 'openmp'))

        # Does it compile? Honoring the OpenMP specification isn't trivial
        assert op.cfunction

        # Does it produce the right result
        op.apply(t_M=9)
        assert np.all(u.data[0] == 10)

        iterations = FindNodes(Iteration).visit(op._func_table['bf0'])
        assert iterations[0].pragmas[0].value == 'omp for collapse(2) schedule(static,1)'
        assert iterations[2].pragmas[0].value ==\
            ('omp parallel for collapse(2) schedule(static,1) num_threads(%d)'
             % nhyperthreads())

    @patch("devito.dse.rewriters.AdvancedRewriter.MIN_COST_ALIAS", 1)
    @patch("devito.dle.parallelizer.Ompizer.NESTED", 0)
    @patch("devito.dle.parallelizer.Ompizer.COLLAPSE", 10000)
    def test_multiple_subnests(self):
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        t = grid.stepping_dim

        f = Function(name='f', grid=grid)
        u = TimeFunction(name='u', grid=grid)

        eqn = Eq(u.forward, ((u[t, x, y, z] + u[t, x+1, y+1, z+1])*3*f +
                             (u[t, x+2, y+2, z+2] + u[t, x+3, y+3, z+3])*3*f + 1))
        op = Operator(eqn, dse='aggressive', dle=('advanced', {'openmp': True}))

        trees = retrieve_iteration_tree(op._func_table['bf0'].root)
        assert len(trees) == 2

        assert trees[0][0] is trees[1][0]
        assert trees[0][0].pragmas[0].value ==\
            'omp for collapse(1) schedule(static,1)'
        assert trees[0][2].pragmas[0].value ==\
            ('omp parallel for collapse(1) schedule(static,1) num_threads(%d)'
             % nhyperthreads())
        assert trees[1][2].pragmas[0].value ==\
            ('omp parallel for collapse(1) schedule(static,1) num_threads(%d)'
             % nhyperthreads())


@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking, _ = _new_operator1(shape, dle='noop')
    w_blocking, _ = _new_operator1(shape, dle='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()
