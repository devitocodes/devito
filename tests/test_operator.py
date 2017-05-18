from __future__ import absolute_import

from conftest import dims

import numpy as np
import pytest
from sympy import Eq  # noqa

from devito import (clear_cache, Operator, DenseData, TimeData,
                    time, t, x, y, z)
from devito.dle import retrieve_iteration_tree
from devito.visitors import IsPerfectIteration


def dimify(dimensions):
    assert isinstance(dimensions, str)
    mapper = dims()
    return tuple(mapper[i] for i in dimensions.split())


def symbol(name, dimensions, value=0., mode='function'):
    """Short-cut for symbol creation to test "function"
    and "indexed" API."""
    assert(mode in ['function', 'indexed'])
    s = DenseData(name=name, dimensions=dimensions)
    s.data[:] = value
    return s.indexify() if mode == 'indexed' else s


class TestArithmetic(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a, a + b + 5.)', 10.),
        ('Eq(a, b - a)', 1.),
        ('Eq(a, 4 * (b * a))', 24.),
        ('Eq(a, (6. / b) + (8. * a))', 18.),
    ])
    @pytest.mark.parametrize('mode', ['function', 'indexed'])
    def test_flat(self, expr, result, mode):
        """Tests basic point-wise arithmetic on two-dimensional data"""
        i, j = dimify('i j')
        a = symbol(name='a', dimensions=(i, j), value=2., mode=mode)
        b = symbol(name='b', dimensions=(i, j), value=3., mode=mode)
        fa = a.base.function if mode == 'indexed' else a
        fb = b.base.function if mode == 'indexed' else b

        eqn = eval(expr)
        Operator(eqn)(fa, fb)
        assert np.allclose(fa.data, result, rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a, a + b + 5.)', 10.),
        ('Eq(a, b - a)', 1.),
        ('Eq(a, 4 * (b * a))', 24.),
        ('Eq(a, (6. / b) + (8. * a))', 18.),
    ])
    @pytest.mark.parametrize('mode', ['function', 'indexed'])
    def test_deep(self, expr, result, mode):
        """Tests basic point-wise arithmetic on multi-dimensional data"""
        i, j, k, l = dimify('i j k l')
        a = symbol(name='a', dimensions=(i, j, k, l), value=2., mode=mode)
        b = symbol(name='b', dimensions=(j, k), value=3., mode=mode)
        fa = a.base.function if mode == 'indexed' else a
        fb = b.base.function if mode == 'indexed' else b

        eqn = eval(expr)
        Operator(eqn)(fa, fb)
        assert np.allclose(fa.data, result, rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[j, l], a[j - 1 , l] + 1.)',
         np.meshgrid(np.arange(2., 8.), np.arange(2., 7.))[1]),
        ('Eq(a[j, l], a[j, l - 1] + 1.)',
         np.meshgrid(np.arange(2., 8.), np.arange(2., 7.))[0]),
    ])
    def test_indexed_increment(self, expr, result):
        """Tests point-wise increments with stencil offsets in one dimension"""
        j, l = dimify('j l')
        a = symbol(name='a', dimensions=(j, l), value=2., mode='indexed').base
        fa = a.function
        fa.data[1:, 1:] = 0

        eqn = eval(expr)
        Operator(eqn)(fa)
        assert np.allclose(fa.data, result, rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[j, l], b[j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[j, l], b[j , l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[j, l], b[j - 1, l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[j, l], b[j + 1, l + 1] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_stencil(self, expr, result):
        """Test point-wise arithmetic with stencil offsets across two
        functions in indexed expression format"""
        j, l = dimify('j l')
        a = symbol(name='a', dimensions=(j, l), value=0., mode='indexed').base
        fa = a.function
        b = symbol(name='b', dimensions=(j, l), value=2., mode='indexed').base
        fb = b.function

        eqn = eval(expr)
        Operator(eqn)(fa, fb)
        assert np.allclose(fa.data[1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[1, j, l], a[0, j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[1, j, l], a[0, j , l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[1, j, l], a[0, j - 1, l - 1] + 1.)', np.zeros((5, 6)) + 3.),
        ('Eq(a[1, j, l], a[0, j + 1, l + 1] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_buffered(self, expr, result):
        """Test point-wise arithmetic with stencil offsets across a single
        functions with buffering dimension in indexed expression format"""
        i, j, l = dimify('i j l')
        a = symbol(name='a', dimensions=(i, j, l), value=2., mode='indexed').base
        fa = a.function

        eqn = eval(expr)
        Operator(eqn)(fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[1, j, l], a[0, j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_open_loops(self, expr, result):
        """Test point-wise arithmetic with stencil offsets and open loop
        boundaries in indexed expression format"""
        i, j, l = dimify('i j l')
        pushed = [d.size for d in [j, l]]
        j.size = None
        l.size = None
        a = DenseData(name='a', dimensions=(i, j, l), shape=(3, 5, 6)).indexed
        fa = a.function
        fa.data[0, :, :] = 2.

        eqn = eval(expr)
        Operator(eqn)(fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)
        j.size, l.size = pushed


class TestArguments(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_override(self):
        """Test that the call-time overriding of Operator arguments works"""
        i, j, k, l = dimify('i j k l')
        a = symbol(name='a', dimensions=(i, j, k, l), value=2.,
                   mode='indexed').base.function
        a1 = symbol(name='a', dimensions=(i, j, k, l), value=3.,
                    mode='indexed').base.function
        a2 = symbol(name='a', dimensions=(i, j, k, l), value=4.,
                    mode='indexed').base.function
        eqn = Eq(a, a+3)
        op = Operator(eqn)
        op()
        op(a=a1)
        op(a=a2)
        shape = [d.size for d in [i, j, k, l]]

        assert(np.allclose(a.data, np.zeros(shape) + 5))
        assert(np.allclose(a1.data, np.zeros(shape) + 6))
        assert(np.allclose(a2.data, np.zeros(shape) + 7))

    def test_dimension_size_infer(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        i, j, k = dimify('i j k')
        shape = tuple([d.size for d in [i, j, k]])
        a = DenseData(name='a', shape=shape).indexed
        b = TimeData(name='b', shape=shape, save=False, time_dim=nt).indexed
        c = TimeData(name='c', shape=shape, save=True, time_dim=nt).indexed
        eqn1 = Eq(b[t, x, y, z], a[x, y, z])
        eqn2 = Eq(c[time, x, y, z], a[x, y, z])
        op1 = Operator(eqn1)
        op2 = Operator(eqn2)

        _, op1_dim_sizes = op1.arguments()
        _, op2_dim_sizes = op2.arguments()
        assert(op1_dim_sizes[time] == 2)
        assert(op2_dim_sizes[time] == nt)


class TestDeclarator(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_heap_1D_stencil(self, a, b):
        operator = Operator(Eq(a, a + b + 5.), dse='noop', dle='noop')
        assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  struct timeval start_loop_i_0, end_loop_i_0;
  gettimeofday(&start_loop_i_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
  }
  gettimeofday(&end_loop_i_0, NULL);
  timings->loop_i_0 += (double)(end_loop_i_0.tv_sec-start_loop_i_0.tv_sec)\
+(double)(end_loop_i_0.tv_usec-start_loop_i_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)

    def test_heap_perfect_2D_stencil(self, a, c):
        operator = Operator([Eq(a, c), Eq(c, c*a)], dse='noop', dle='noop', aaa=True)
        assert """\
  float (*a);
  float (*c)[5];
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  posix_memalign((void**)&c, 64, sizeof(float[3][5]));
  struct timeval start_loop_i_0, end_loop_i_0;
  gettimeofday(&start_loop_i_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      a[i] = c[i][j];
      c[i][j] = a[i]*c[i][j];
    }
  }
  gettimeofday(&end_loop_i_0, NULL);
  timings->loop_i_0 += (double)(end_loop_i_0.tv_sec-start_loop_i_0.tv_sec)\
+(double)(end_loop_i_0.tv_usec-start_loop_i_0.tv_usec)/1000000;
  free(a);
  free(c);
  return 0;""" in str(operator.ccode)

    def test_heap_imperfect_2D_stencil(self, a, c):
        operator = Operator([Eq(a, 0.), Eq(c, c*a)], dse='noop', dle='noop')
        assert """\
  float (*a);
  float (*c)[5];
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  posix_memalign((void**)&c, 64, sizeof(float[3][5]));
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = 0.0F;
    struct timeval start_loop_j_0, end_loop_j_0;
    gettimeofday(&start_loop_j_0, NULL);
    for (int j = 0; j < 5; j += 1)
    {
      c[i][j] = a[i]*c[i][j];
    }
    gettimeofday(&end_loop_j_0, NULL);
    timings->loop_j_0 += (double)(end_loop_j_0.tv_sec-start_loop_j_0.tv_sec)\
+(double)(end_loop_j_0.tv_usec-start_loop_j_0.tv_usec)/1000000;
  }
  free(a);
  free(c);
  return 0;""" in str(operator.ccode)

    def test_stack_scalar_temporaries(self, a, t0, t1):
        operator = Operator([Eq(t0, 1.), Eq(t1, 2.), Eq(a, t0*t1*3.)],
                            dse='noop', dle='noop')
        assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  struct timeval start_loop_i_0, end_loop_i_0;
  gettimeofday(&start_loop_i_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    float t0 = 1.00000000000000F;
    float t1 = 2.00000000000000F;
    a[i] = 3.0F*t0*t1;
  }
  gettimeofday(&end_loop_i_0, NULL);
  timings->loop_i_0 += (double)(end_loop_i_0.tv_sec-start_loop_i_0.tv_sec)\
+(double)(end_loop_i_0.tv_usec-start_loop_i_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)

    def test_stack_vector_temporaries(self, c_stack, e):
        operator = Operator([Eq(c_stack, e*1.)], dse='noop', dle='noop')
        assert """\
  struct timeval start_loop_k_0, end_loop_k_0;
  gettimeofday(&start_loop_k_0, NULL);
  for (int k = 0; k < 7; k += 1)
  {
    for (int s = 0; s < 4; s += 1)
    {
      for (int q = 0; q < 4; q += 1)
      {
        double c_stack[3][5] __attribute__((aligned(64)));
        for (int i = 0; i < 3; i += 1)
        {
          for (int j = 0; j < 5; j += 1)
          {
            c_stack[i][j] = 1.0F*e[k][s][q][i][j];
          }
        }
      }
    }
  }
  gettimeofday(&end_loop_k_0, NULL);
  timings->loop_k_0 += (double)(end_loop_k_0.tv_sec-start_loop_k_0.tv_sec)\
+(double)(end_loop_k_0.tv_usec-start_loop_k_0.tv_usec)/1000000;
  return 0;""" in str(operator.ccode)


class TestLoopScheduler(object):

    def test_consistency_perfect_loops(self, tu, tv, ti0, t0, t1):
        eq1 = Eq(tu, tv*ti0*t0 + ti0*t1)
        eq2 = Eq(ti0, tu + t0*3.)
        eq3 = Eq(tv, ti0*tu)
        op1 = Operator([eq1, eq2, eq3], dse='noop', dle='noop')
        op2 = Operator([eq2, eq1, eq3], dse='noop', dle='noop')
        op3 = Operator([eq3, eq2, eq1], dse='noop', dle='noop')

        trees = [retrieve_iteration_tree(i) for i in [op1, op2, op3]]
        assert all(len(i) == 1 for i in trees)
        trees = [i[0] for i in trees]
        for tree in trees:
            assert IsPerfectIteration().visit(tree[0])
            assert len(tree[-1].nodes) == 3
        pivot = set([j.expr for j in trees[0][-1].nodes])
        assert all(set([j.expr for j in i[-1].nodes]) == pivot for i in trees)

    def test_expressions_imperfect_loops(self, ti0, ti1, ti2, t0):
        eq1 = Eq(ti2, t0*3.)
        eq2 = Eq(ti0, ti1 + 4. + ti2*5.)
        op = Operator([eq1, eq2], dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        outer, inner = trees
        assert len(outer) == 2 and len(inner) == 3
        assert all(i == j for i, j in zip(outer, inner[:-1]))
        assert outer[-1].nodes[0].expr.rhs == eq1.rhs
        assert inner[-1].nodes[0].expr.rhs == eq2.rhs

    def test_different_loop_nests(self, tu, ti0, t0, t1):
        eq1 = Eq(ti0, t0*3.)
        eq2 = Eq(tu, ti0 + t1*3.)
        op = Operator([eq1, eq2], dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].expr.rhs == eq1.rhs
        assert trees[1][-1].nodes[0].expr.rhs == eq2.rhs
