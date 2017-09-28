from __future__ import absolute_import

from collections import OrderedDict

from conftest import EVAL, dims, dims_open

import numpy as np
import pytest

from devito import (clear_cache, Operator, ConstantData, DenseData, TimeData,
                    PointData, Dimension, Eq, time, x, y, z, configuration)
from devito.foreign import Operator as OperatorForeign
from devito.dle import retrieve_iteration_tree
from devito.visitors import IsPerfectIteration


def dimify(dimensions, open=False):
    assert isinstance(dimensions, str)
    mapper = dims() if not open else dims_open()
    return tuple(mapper[i] for i in dimensions.split())


def symbol(name, dimensions, value=0., mode='function'):
    """Short-cut for symbol creation to test "function"
    and "indexed" API."""
    assert(mode in ['function', 'indexed'])
    s = DenseData(name=name, dimensions=dimensions)
    s.data[:] = value
    return s.indexify() if mode == 'indexed' else s


class TestAPI(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_code_generation(self, const, a_dense):
        """
        Tests that we can actually generate code for a trivial operator
        using constant and array data objects.
        """
        eqn = Eq(a_dense, a_dense + 2.*const)
        op = Operator(eqn)
        assert len(op.parameters) == 3
        assert op.parameters[0].name == 'a_dense'
        assert op.parameters[0].is_TensorArgument
        assert op.parameters[1].name == 'constant'
        assert op.parameters[1].is_ScalarArgument
        assert op.parameters[2].name == 'timings'
        assert op.parameters[2].is_PtrArgument
        assert 'a_dense[i] = 2.0F*constant + a_dense[i]' in str(op.ccode)


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
        Operator(eqn)(a=fa, b=fb)
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
        Operator(eqn)(a=fa, b=fb)
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
        Operator(eqn)(a=fa)
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
        Operator(eqn)(a=fa, b=fb)
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
        Operator(eqn)(a=fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    @pytest.mark.parametrize('expr, result', [
        ('Eq(a[1, j, l], a[0, j - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
    ])
    def test_indexed_open_loops(self, expr, result):
        """Test point-wise arithmetic with stencil offsets and open loop
        boundaries in indexed expression format"""
        i, j, l = dimify('i j l', open=True)
        a = DenseData(name='a', dimensions=(i, j, l), shape=(3, 5, 6)).indexed
        fa = a.function
        fa.data[0, :, :] = 2.

        eqn = eval(expr)
        Operator(eqn)(a=fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    def test_constant_time_dense(self):
        """Test arithmetic between different data objects, namely ConstantData
        and DenseData."""
        const = ConstantData(name='truc', value=2.)
        a = DenseData(name='a', shape=(20, 20))
        a.data[:] = 2.
        eqn = Eq(a, a + 2.*const)
        op = Operator(eqn)
        op.apply(a=a, truc=const)
        assert(np.allclose(a.data, 6.))

        # Applying a different constant still works
        op.apply(a=a, truc=ConstantData(name='truc2', value=3.))
        assert(np.allclose(a.data, 12.))


class TestArguments(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_override_cache_aliasing(self):
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

    def test_override_symbol(self):
        """Test call-time symbols overrides with other symbols"""
        i, j, k, l = dimify('i j k l')
        a = symbol(name='a', dimensions=(i, j, k, l), value=2.)
        a1 = symbol(name='a1', dimensions=(i, j, k, l), value=3.)
        a2 = symbol(name='a2', dimensions=(i, j, k, l), value=4.)
        op = Operator(Eq(a, a + 3))
        op()
        op(a=a1)
        op(a=a2)
        shape = [d.size for d in [i, j, k, l]]

        assert(np.allclose(a.data, np.zeros(shape) + 5))
        assert(np.allclose(a1.data, np.zeros(shape) + 6))
        assert(np.allclose(a2.data, np.zeros(shape) + 7))

    def test_override_array(self):
        """Test call-time symbols overrides with numpy arrays"""
        i, j, k, l = dimify('i j k l')
        shape = tuple(d.size for d in (i, j, k, l))
        a = symbol(name='a', dimensions=(i, j, k, l), value=2.)
        a1 = np.zeros(shape=shape, dtype=np.float32) + 3.
        a2 = np.zeros(shape=shape, dtype=np.float32) + 4.
        op = Operator(Eq(a, a + 3))
        op()
        op(a=a1)
        op(a=a2)
        shape = [d.size for d in [i, j, k, l]]

        assert(np.allclose(a.data, np.zeros(shape) + 5))
        assert(np.allclose(a1, np.zeros(shape) + 6))
        assert(np.allclose(a2, np.zeros(shape) + 7))

    def test_dimension_size_infer(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        i, j, k = dimify('i j k')
        shape = tuple([d.size for d in [i, j, k]])
        a = DenseData(name='a', shape=shape).indexed
        b = TimeData(name='b', shape=shape, save=True, time_dim=nt).indexed
        eqn = Eq(b[time, x, y, z], a[x, y, z])
        op = Operator(eqn)

        _, op_dim_sizes = op.arguments()
        assert(op_dim_sizes[time.name] == nt)

    def test_dimension_size_override(self, nt=100):
        """Test explicit overrides for the leading time dimension"""
        i, j, k = dimify('i j k')
        a = TimeData(name='a', dimensions=(i, j, k))
        one = symbol(name='one', dimensions=(i, j, k), value=1.)
        op = Operator(Eq(a.forward, a + one))

        # Test dimension override via the buffered dimenions
        a.data[0] = 0.
        op(a=a, t=6)
        assert(np.allclose(a.data[1], 5.))

        # Test dimension override via the parent dimenions
        a.data[0] = 0.
        op(a=a, time=5)
        assert(np.allclose(a.data[0], 4.))

    def test_override_composite_data(self):
        original_coords = (1., 1.)
        new_coords = (2., 2.)
        p_dim = Dimension('p_src')
        ndim = len(original_coords)
        u = TimeData(name='u', time_order=2, space_order=2, shape=(10, 10))
        src1 = PointData(name='src1', dimensions=[time, p_dim], npoint=1, nt=10,
                         ndim=ndim, coordinates=original_coords)
        src2 = PointData(name='src1', dimensions=[time, p_dim], npoint=1, nt=10,
                         ndim=ndim, coordinates=new_coords)
        op = Operator(src1.inject(u, src1))

        # Move the source from the location where the setup put it so we can test
        # whether the override picks up the original coordinates or the changed ones

        # Operator.arguments() returns a tuple of (data, dimension_sizes)
        args = op.arguments(src1=src2)[0]
        arg_name = src1.name + "_coords"
        assert(np.array_equal(args[arg_name], np.asarray((new_coords,))))


class TestDeclarator(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_heap_1D_stencil(self, a, b):
        operator = Operator(Eq(a, a + b + 5.), dse='noop', dle='noop')
        assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
  }
  gettimeofday(&end_section_0, NULL);
  timings->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)

    def test_heap_perfect_2D_stencil(self, a, c):
        operator = Operator([Eq(a, c), Eq(c, c*a)], dse='noop', dle='noop', aaa=True)
        assert """\
  float (*a);
  float (*c)[5];
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  posix_memalign((void**)&c, 64, sizeof(float[3][5]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      a[i] = c[i][j];
      c[i][j] = a[i]*c[i][j];
    }
  }
  gettimeofday(&end_section_0, NULL);
  timings->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
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
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    a[i] = 0.0F;
    for (int j = 0; j < 5; j += 1)
    {
      c[i][j] = a[i]*c[i][j];
    }
  }
  gettimeofday(&end_section_0, NULL);
  timings->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(a);
  free(c);
  return 0;""" in str(operator.ccode)

    def test_stack_scalar_temporaries(self, a, t0, t1):
        operator = Operator([Eq(t0, 1.), Eq(t1, 2.), Eq(a, t0*t1*3.)],
                            dse='noop', dle='noop')
        assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[3]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = 0; i < 3; i += 1)
  {
    float t0 = 1.00000000000000F;
    float t1 = 2.00000000000000F;
    a[i] = 3.0F*t0*t1;
  }
  gettimeofday(&end_section_0, NULL);
  timings->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)

    def test_stack_vector_temporaries(self, c_stack, e):
        operator = Operator([Eq(c_stack, e*1.)], dse='noop', dle='noop')
        assert """\
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int k = 0; k < 7; k += 1)
  {
    float c_stack[3][5] __attribute__((aligned(64)));
    for (int s = 0; s < 4; s += 1)
    {
      for (int q = 0; q < 4; q += 1)
      {
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
  gettimeofday(&end_section_0, NULL);
  timings->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  return 0;""" in str(operator.ccode)


class TestLoopScheduler(object):

    def test_consistency_coupled_wo_ofs(self, tu, tv, ti0, t0, t1):
        """
        Test that no matter what is the order in which the equations are
        provided to an Operator, the resulting loop nest is the same.
        None of the array accesses in the equations use offsets.
        """
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

    @pytest.mark.parametrize('exprs', [
        ('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])', 'Eq(ti1[x,y,z], ti3[x,y,z])',
         'Eq(ti3[x,y,z], ti1[x,y,z] + 1.)'),
        ('Eq(ti0[x,y,z], ti0[x,y,z-1] + ti1[x,y,z+1])', 'Eq(ti1[x,y,z], ti3[x,y,z+1])',
         'Eq(ti3[x,y,z], ti3[x,y,z-1] + ti0[x,y,z])'),
        ('Eq(ti0[x,y,z+2], ti0[x,y,z-1] + ti1[x,y,z+1])',
         'Eq(ti1[x,y,z+3], ti3[x,y,z+1])',
         'Eq(ti3[x,y,z+2], ti0[x,y,z+1]*ti3[x,y,z-1])'),
        ('Eq(ti0[x,y,z], ti0[x-2,y-1,z-1] + ti1[x+2,y+3,z+1])',
         'Eq(ti1[x,y,z], ti3[x+1,y-4,z+1])',
         'Eq(ti3[x,y,z], ti3[x+5,y,z-1] - ti0[x+3,y-2,z+4])')
    ])
    def test_consistency_coupled_w_ofs(self, exprs, ti0, ti1, ti3):
        """
        Test that no matter what is the order in which the equations are
        provided to an Operator, the resulting loop nest is the same.
        The array accesses in the equations may or may not use offsets;
        these impact the loop bounds, but not the resulting tree
        structure.
        """
        eq1, eq2, eq3 = EVAL(exprs, ti0.base, ti1.base, ti3.base)
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
        """
        Test that equations depending only on a subset of all indices
        appearing across all equations are placed within earlier loops
        in the loop nest tree.
        """
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

    def test_equations_emulate_bc(self, t0):
        """
        Test that bc-like equations get inserted into the same loop nest
        as the "main" equations.
        """
        shape = (3, 3, 3)
        a = DenseData(name='a', shape=shape).indexed
        b = TimeData(name='b', shape=shape, save=True, time_dim=6).indexed
        main = Eq(b[time + 1, x, y, z], b[time - 1, x, y, z] + a[x, y, z] + 3.*t0)
        bcs = [Eq(b[time, 0, y, z], 0.),
               Eq(b[time, x, 0, z], 0.),
               Eq(b[time, x, y, 0], 0.)]
        op = Operator([main] + bcs, dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 4
        assert all(id(trees[0][0]) == id(i[0]) for i in trees)

    def test_different_section_nests(self, tu, ti0, t0, t1):
        eq1 = Eq(ti0, t0*3.)
        eq2 = Eq(tu, ti0 + t1*3.)
        op = Operator([eq1, eq2], dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].expr.rhs == eq1.rhs
        assert trees[1][-1].nodes[0].expr.rhs == eq2.rhs

    @pytest.mark.parametrize('exprs', [
        ['Eq(ti0[x,y,z], ti0[x,y,z] + t0*2.)', 'Eq(ti0[0,0,z], 0.)'],
        ['Eq(ti0[x,y,z], ti0[x,y,z-1] + t0*2.)', 'Eq(ti0[0,0,z], 0.)'],
        ['Eq(ti0[x,y,z], ti0[x,y,z] + t0*2.)', 'Eq(ti0[0,y,0], 0.)'],
        ['Eq(ti0[x,y,z], ti0[x,y,z] + t0*2.)', 'Eq(ti0[0,y,z], 0.)'],
    ])
    def test_directly_indexed_expression(self, fa, ti0, t0, exprs):
        """
        Test that equations using integer indices are inserted in the right
        loop nest, at the right loop nest depth.
        """
        eqs = EVAL(exprs, ti0.base, t0)
        op = Operator(eqs, dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert trees[0][-1].nodes[0].expr.rhs == eqs[0].rhs
        assert trees[1][-1].nodes[0].expr.rhs == eqs[1].rhs

    @pytest.mark.parametrize('shape, dimensions', [((11, 11), (x, y)),
                                                   ((11, 11), (y, x)),
                                                   ((11, 11, 11), (x, y, z)),
                                                   ((11, 11, 11), (x, z, y)),
                                                   ((11, 11, 11), (y, x, z)),
                                                   ((11, 11, 11), (y, z, x)),
                                                   ((11, 11, 11), (z, x, y)),
                                                   ((11, 11, 11), (z, y, x))])
    def test_equations_mixed_densedata_timedata(self, shape, dimensions):
        """
        Test that equations using a mixture of DenseData and TimeData objects
        are embedded within the same time loop.
        """
        a = TimeData(name='a', shape=shape, time_order=2, dimensions=dimensions,
                     space_order=2, time_dim=6, save=False)
        p_aux = Dimension(name='p_aux', size=10)
        b = DenseData(name='b', shape=shape + (10,), dimensions=dimensions + (p_aux,),
                      space_order=2)
        b.data[:] = 1.0
        b2 = DenseData(name='b2', shape=(10,) + shape, dimensions=(p_aux,) + dimensions,
                       space_order=2)
        b2.data[:] = 1.0
        eqns = [Eq(a.forward, a.laplace + 1.),
                Eq(b, time*b*a + b)]
        eqns2 = [Eq(a.forward, a.laplace + 1.),
                 Eq(b2, time*b2*a + b2)]
        subs = {x.spacing: 2.5, y.spacing: 1.5, z.spacing: 2.0}
        op = Operator(eqns, subs=subs, dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert all(trees[0][i] is trees[1][i] for i in range(3))

        op2 = Operator(eqns2, subs=subs, dle='noop')
        trees = retrieve_iteration_tree(op2)
        assert len(trees) == 2
        assert all(trees[0][i] is trees[1][i] for i in range(3))

        # Verify both operators produce the same result
        op()
        op2()

        assert(np.allclose(b2.data[2, ...].reshape(-1) -
                           b.data[..., 2].reshape(-1), 0.))


@pytest.mark.skipif(configuration['backend'] != 'foreign',
                    reason="'foreign' wasn't selected as backend on startup")
class TestForeign(object):

    def test_explicit_run(self):
        time_dim = 6
        a = TimeData(name='a', shape=(11, 11), time_order=1,
                     time_dim=time_dim, save=True)
        eqn = Eq(a.forward, a + 1.)
        op = Operator(eqn)
        assert isinstance(op, OperatorForeign)
        args = OrderedDict(op.arguments())
        assert args['a'] is None
        # Emulate data feeding from outside
        array = np.ndarray(shape=a.shape, dtype=np.float32)
        array.fill(0.0)
        args['a'] = array
        op.cfunction(*list(args.values()))
        assert all(np.allclose(args['a'][i], i) for i in range(time_dim))
