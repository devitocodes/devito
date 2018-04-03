from __future__ import absolute_import

from functools import reduce
from operator import mul
import numpy as np
import pytest
from conftest import skipif_yask
from sympy import solve

from conftest import EVAL

from devito.dle import transform
from devito import Grid, Function, TimeFunction, Eq, Operator
from devito.ir.equations import DummyEq
from devito.ir.iet import (ELEMENTAL, Expression, Callable, Iteration, List, tagger,
                           Transformer, FindNodes, iet_analyze, retrieve_iteration_tree)


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


def _new_operator1(shape, **kwargs):
    grid = Grid(shape=shape, dtype=np.int32)
    infield = Function(name='infield', grid=grid)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)

    outfield = Function(name='outfield', grid=grid)

    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)

    # Run the operator
    op = Operator(stencil, **kwargs)
    op(infield=infield, outfield=outfield)

    return outfield, op


def _new_operator2(shape, time_order, **kwargs):
    grid = Grid(shape=shape, dtype=np.int32)
    infield = TimeFunction(name='infield', grid=grid, time_order=time_order)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)

    outfield = TimeFunction(name='outfield', grid=grid, time_order=time_order)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)

    # Run the operator
    op = Operator(stencil, **kwargs)
    op(infield=infield, outfield=outfield, t=10)

    return outfield, op


def _new_operator3(shape, **kwargs):
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
    stencil = solve(eqn, u.forward, rational=False)[0]
    op = Operator(Eq(u.forward, stencil), **kwargs)

    # Execute the generated Devito stencil operator
    op.apply(u=u, t=10, dt=dt)
    return u.data[1, :], op


@skipif_yask
def test_create_elemental_functions_simple(simple_function):
    roots = [i[-1] for i in retrieve_iteration_tree(simple_function)]
    retagged = [i._rebuild(properties=tagger(0)) for i in roots]
    mapper = {i: j._rebuild(properties=(j.properties + (ELEMENTAL,)))
              for i, j in zip(roots, retagged)}
    function = Transformer(mapper).visit(simple_function)
    handle = transform(function, mode='split')
    block = List(body=[handle.nodes] + handle.elemental_functions)
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
      f_0((float*)a,(float*)b,(float*)c,(float*)d,i,j,j_size,7,k_size,0);
    }
  }
}
void f_0(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec,"""
         """ const int i, const int j, const int j_size,"""
         """ const int k_finish, const int k_size, const int k_start)
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


@skipif_yask
def test_create_elemental_functions_complex(complex_function):
    roots = [i[-1] for i in retrieve_iteration_tree(complex_function)]
    retagged = [j._rebuild(properties=tagger(i)) for i, j in enumerate(roots)]
    mapper = {i: j._rebuild(properties=(j.properties + (ELEMENTAL,)))
              for i, j in zip(roots, retagged)}
    function = Transformer(mapper).visit(complex_function)
    handle = transform(function, mode='split')
    block = List(body=[handle.nodes] + handle.elemental_functions)
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
    f_0((float*)a,(float*)b,i,4,0);
    for (int j = 0; j <= 5; j += 1)
    {
      f_1((float*)a,(float*)b,(float*)c,(float*)d,i,j,j_size,7,k_size,0);
    }
    f_2((float*)a,(float*)b,i,4,0);
  }
}
void f_0(float *restrict a_vec, float *restrict b_vec,"""
         """ const int i, const int s_finish, const int s_start)
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
         """ const int i, const int j, const int j_size,"""
         """ const int k_finish, const int k_size, const int k_start)
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
         """ const int i, const int q_finish, const int q_start)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  for (int q = q_start; q <= q_finish; q += 1)
  {
    a[i] = 8.0F*a[i] + 6.0F/b[i];
  }
}""")


@skipif_yask
@pytest.mark.parametrize("blockinner,expected", [
    (False, 4),
    (True, 8)
])
def test_cache_blocking_structure(blockinner, expected):
    _, op = _new_operator1((10, 31, 45), dle=('blocking', {'blockalways': True,
                                                           'blockshape': (2, 9, 2),
                                                           'blockinner': blockinner}))

    # Check presence of remainder loops
    iterations = retrieve_iteration_tree(op)
    assert len(iterations) == expected
    assert not iterations[0][0].is_Remainder
    assert all(i[0].is_Remainder for i in iterations[1:])

    # Check presence of openmp pragmas at the right place
    _, op = _new_operator1((10, 31, 45), dle=('blocking,openmp',
                                              {'blockalways': True,
                                               'blockshape': (2, 9, 2),
                                               'blockinner': blockinner}))
    iterations = retrieve_iteration_tree(op)
    assert len(iterations) == expected
    # All iterations except the last one an outermost parallel loop over blocks
    assert not iterations[-1][0].is_Parallel
    for i in iterations[:-1]:
        outermost = i[0]
        assert len(outermost.pragmas) == 1
        assert 'omp for' in outermost.pragmas[0].value


@skipif_yask
@pytest.mark.parametrize("shape", [(10,), (10, 45), (10, 31, 45)])
@pytest.mark.parametrize("blockshape", [2, 7, (3, 3), (2, 9, 1)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_no_time_loop(shape, blockshape, blockinner):
    wo_blocking, _ = _new_operator1(shape, dle='noop')
    w_blocking, _ = _new_operator1(shape, dle=('blocking', {'blockalways': True,
                                                            'blockshape': blockshape,
                                                            'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@skipif_yask
@pytest.mark.parametrize("shape", [(20, 33), (45, 31, 45)])
@pytest.mark.parametrize("time_order", [2])
@pytest.mark.parametrize("blockshape", [2, (13, 20), (11, 15, 23)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_time_loop(shape, time_order, blockshape, blockinner):
    wo_blocking, _ = _new_operator2(shape, time_order, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order,
                                   dle=('blocking', {'blockshape': blockshape,
                                                     'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@skipif_yask
@pytest.mark.parametrize("shape,blockshape", [
    ((25, 25, 46), (None, None, None)),
    ((25, 25, 46), (7, None, None)),
    ((25, 25, 46), (None, None, 7)),
    ((25, 25, 46), (None, 7, None)),
    ((25, 25, 46), (5, None, 7)),
    ((25, 25, 46), (10, 3, None)),
    ((25, 25, 46), (None, 7, 11)),
    ((25, 25, 46), (8, 2, 4)),
    ((25, 25, 46), (2, 4, 8)),
    ((25, 25, 46), (4, 8, 2)),
    ((25, 46), (None, 7)),
    ((25, 46), (7, None))
])
def test_cache_blocking_edge_cases(shape, blockshape):
    wo_blocking, _ = _new_operator2(shape, time_order=2, dle='noop')
    w_blocking, _ = _new_operator2(shape, time_order=2,
                                   dle=('blocking', {'blockshape': blockshape,
                                                     'blockinner': True}))
    assert np.equal(wo_blocking.data, w_blocking.data).all()


@skipif_yask
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
    w_blocking, b = _new_operator3(shape, dle=('blocking', {'blockshape': blockshape,
                                                            'blockinner': True}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@skipif_yask
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
    # outermost parallel w/ repeated dimensions
    (['Eq(t0, fc[x,x] + fd[x,y+1])', 'Eq(fc[x,x], t0 + 1)'],
     (True, False)),
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


@skipif_yask
@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking, _ = _new_operator1(shape, dle='noop')
    w_blocking, _ = _new_operator1(shape, dle='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()
