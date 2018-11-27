from functools import reduce
from operator import mul
import numpy as np

import pytest
from conftest import EVAL

from devito import Grid, Function, TimeFunction, Eq, Operator, solve, configuration
from devito.dle import transform
from devito.ir.equations import DummyEq
from devito.ir.iet import (ELEMENTAL, Expression, Callable, Iteration, List, tagger,
                           Transformer, FindNodes, iet_analyze, retrieve_iteration_tree)
from devito.tools import as_tuple
from unittest.mock import patch

pytestmark = pytest.mark.skipif(configuration['backend'] == 'yask' or
                                configuration['backend'] == 'ops',
                                reason="testing is currently restricted")


@pytest.fixture(scope="module")
def exprs(a, b, c, d, a_dense, b_dense):
    return [Expression(DummyEq(a, a + b + 5.)),
            Expression(DummyEq(a, b*d - a*c)),
            Expression(DummyEq(b, a + b*b + 3)),
            Expression(DummyEq(a, a*b*d*c)),
            Expression(DummyEq(a, 4 * ((b + d) * (a + c)))),
            Expression(DummyEq(a, (6. / b) + (8. * a))),
            Expression(DummyEq(a_dense, a_dense + b_dense + 5.))]


@pytest.fixture(scope="module")
def simple_function(a, b, c, d, exprs, iters):
    # void foo(a, b)
    #   for i
    #     for j
    #       for k
    #         expr0
    #         expr1
    symbols = [i.base.function for i in [a, b, c, d]]
    body = iters[0](iters[1](iters[2]([exprs[0], exprs[1]])))
    return Callable('foo', body, 'void', symbols, ())


@pytest.fixture(scope="module")
def simple_function_with_paddable_arrays(a_dense, b_dense, exprs, iters):
    # void foo(a_dense, b_dense)
    #   for i
    #     for j
    #       for k
    #         expr0
    symbols = [i.base.function for i in [a_dense, b_dense]]
    body = iters[0](iters[1](iters[2](exprs[6])))
    return Callable('foo', body, 'void', symbols, ())


@pytest.fixture(scope="module")
def complex_function(a, b, c, d, exprs, iters):
    # void foo(a, b, c, d)
    #   for i
    #     for s
    #       expr0
    #     for j
    #       for k
    #         expr1
    #         expr2
    #     for p
    #       expr3
    symbols = [i.base.function for i in [a, b, c, d]]
    body = iters[0]([iters[3](exprs[2]),
                     iters[1](iters[2]([exprs[3], exprs[4]])),
                     iters[4](exprs[5])])
    return Callable('foo', body, 'void', symbols, ())


def get_blocksizes(op, dle, grid, blockshape):
    blocksizes = {'%s0_block_size' % d: v for d, v in zip(grid.dimensions, blockshape)}
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


def _new_operator3(shape, blockshape=None, dle=None):
    blockshape = as_tuple(blockshape)
    grid = Grid(shape=shape)
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

    blocksizes = get_blocksizes(op, dle, grid, blockshape)
    op.apply(u=u, t=10, dt=dt, **blocksizes)

    return u.data[1, :], op


def test_create_efuncs_simple(simple_function):
    roots = [i[-1] for i in retrieve_iteration_tree(simple_function)]
    retagged = [i._rebuild(properties=tagger(0)) for i in roots]
    mapper = {i: j._rebuild(properties=(j.properties + (ELEMENTAL,)))
              for i, j in zip(roots, retagged)}
    function = Transformer(mapper).visit(simple_function)
    handle = transform(function, mode='split')
    block = List(body=[handle.nodes] + handle.efuncs)
    output = str(block.ccode)
    # Make output compiler independent
    output = [i for i in output.split('\n')
              if all([j not in i for j in ('#pragma', '/*')])]
    assert '\n'.join(output) == \
        ("""void foo(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec)
{
  for (int i = 0; i <= 3; i += 1)
  {
    for (int j = 0; j <= 5; j += 1)
    {
      f_0((float*)a,(float*)b,(float*)c,(float*)d,i_size,j_size,k_size,i,j,7,0);
    }
  }
}
void f_0(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec,"""
         """ const int i_size, const int j_size, const int k_size,"""
         """ const int i, const int j, const int k_finish, const int k_start)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  float (*restrict c)[j_size] __attribute__((aligned(64))) = (float (*)[j_size]) c_vec;
  float (*restrict d)[j_size][k_size] __attribute__((aligned(64))) ="""
         """ (float (*)[j_size][k_size]) d_vec;
  for (int k = k_start; k <= k_finish; k += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
    a[i] = -a[i]*c[i][j] + b[i]*d[i][j][k];
  }
}""")


def test_create_efuncs_complex(complex_function):
    roots = [i[-1] for i in retrieve_iteration_tree(complex_function)]
    retagged = [j._rebuild(properties=tagger(i)) for i, j in enumerate(roots)]
    mapper = {i: j._rebuild(properties=(j.properties + (ELEMENTAL,)))
              for i, j in zip(roots, retagged)}
    function = Transformer(mapper).visit(complex_function)
    handle = transform(function, mode='split')
    block = List(body=[handle.nodes] + handle.efuncs)
    output = str(block.ccode)
    # Make output compiler independent
    output = [i for i in output.split('\n')
              if all([j not in i for j in ('#pragma', '/*')])]
    assert '\n'.join(output) == \
        ("""void foo(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec)
{
  for (int i = 0; i <= 3; i += 1)
  {
    f_0((float*)a,(float*)b,i_size,i,4,0);
    for (int j = 0; j <= 5; j += 1)
    {
      f_1((float*)a,(float*)b,(float*)c,(float*)d,i_size,j_size,k_size,i,j,7,0);
    }
    f_2((float*)a,(float*)b,i_size,i,4,0);
  }
}
void f_0(float *restrict a_vec, float *restrict b_vec,"""
         """ const int i_size, const int i, const int s_finish, const int s_start)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  for (int s = s_start; s <= s_finish; s += 1)
  {
    b[i] = a[i] + pow(b[i], 2) + 3;
  }
}
void f_1(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec,"""
         """ const int i_size, const int j_size, const int k_size,"""
         """ const int i, const int j, const int k_finish, const int k_start)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  float (*restrict c)[j_size] __attribute__((aligned(64))) = (float (*)[j_size]) c_vec;
  float (*restrict d)[j_size][k_size] __attribute__((aligned(64))) ="""
         """ (float (*)[j_size][k_size]) d_vec;
  for (int k = k_start; k <= k_finish; k += 1)
  {
    a[i] = a[i]*b[i]*c[i][j]*d[i][j][k];
    a[i] = 4*(a[i] + c[i][j])*(b[i] + d[i][j][k]);
  }
}
void f_2(float *restrict a_vec, float *restrict b_vec,"""
         """ const int i_size, const int i, const int q_finish, const int q_start)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  for (int q = q_start; q <= q_finish; q += 1)
  {
    a[i] = 8.0F*a[i] + 6.0F/b[i];
  }
}""")


@pytest.mark.parametrize("blockinner,expected", [
    (False, 4),
    (True, 8)
])
def test_cache_blocking_structure(blockinner, expected):
    _, op = _new_operator1((10, 31, 45), dle=('blocking', {'blockalways': True,
                                                           'blockinner': blockinner}))

    # Check presence of remainder loops
    iterations = retrieve_iteration_tree(op)
    assert len(iterations) == expected
    assert not iterations[0][0].is_Remainder
    assert all(i[0].is_Remainder for i in iterations[1:])

    # Check presence of openmp pragmas at the right place
    _, op = _new_operator1((10, 31, 45), dle=('blocking',
                                              {'openmp': True,
                                               'blockalways': True,
                                               'blockinner': blockinner}))
    iterations = retrieve_iteration_tree(op)
    assert len(iterations) == expected
    # All iterations except the last one an outermost parallel loop over blocks
    assert not iterations[-1][0].is_Parallel
    for i in iterations[:-1]:
        outermost = i[0]
        assert len(outermost.pragmas) == 1
        assert 'omp for' in outermost.pragmas[0].value


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
    ((3, 3), (3, 4)),
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

    assert np.equal(wo_blocking.data, w_blocking.data).all()


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
def test_loops_ompized(fa, fb, fc, fd, t0, t1, t2, t3, exprs, expected, iters):
    scope = [fa, fb, fc, fd, t0, t1, t2, t3]
    node_exprs = [Expression(DummyEq(EVAL(i, *scope))) for i in exprs]
    ast = iters[6](iters[7](node_exprs))

    ast = iet_analyze(ast)

    nodes = transform(ast, mode='openmp').nodes
    iterations = FindNodes(Iteration).visit(nodes)
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


def test_dynamic_nthreads():
    grid = Grid(shape=(16, 16, 16))
    f = TimeFunction(name='f', grid=grid)

    op = Operator(Eq(f.forward, f + 1.), dle='openmp')

    # Check num_threads appears in the generated code
    # Not very elegant, but it does the trick
    assert 'num_threads(nt)' in str(op)

    # Check `op` accepts the `nthreads` kwarg
    op.apply(time=0)
    op.apply(time_m=1, time_M=1, nthreads=4)
    assert np.all(f.data[0] == 2.)

    # Check the actual value assumed by `nthreads`
    from devito.dle.backends.parallelizer import ncores
    assert op.arguments(time=0)['nthreads'] == ncores()  # default value
    assert op.arguments(time=0, nthreads=123)['nthreads'] == 123  # user supplied


@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking, _ = _new_operator1(shape, dle='noop')
    w_blocking, _ = _new_operator1(shape, dle='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize('exprs,expected', [
    # trivial 1D
    (['Eq(fe[x,y,z], fe[x,y,z] + fe[x,y,z])'],
     (True, False, False))
])
@patch("devito.dle.backends.parallelizer.Ompizer.COLLAPSE", 1)
def test_loops_collapsed(fe, t0, t1, t2, t3, exprs, expected, iters):
    scope = [fe, t0, t1, t2, t3]
    node_exprs = [Expression(DummyEq(EVAL(i, *scope))) for i in exprs]
    ast = iters[6](iters[7](iters[8](node_exprs)))

    ast = iet_analyze(ast)

    nodes = transform(ast, mode='openmp').nodes
    iterations = FindNodes(Iteration).visit(nodes)
    assert len(iterations) == len(expected)

    # Check for presence of pragma omp
    for i, j in zip(iterations, expected):
        pragmas = i.pragmas
        if j is True:
            assert len(pragmas) == 1
            pragma = pragmas[0]
            assert 'omp for collapse' in pragma.value
        else:
            for k in pragmas:
                assert 'omp for collapse' not in k.value
