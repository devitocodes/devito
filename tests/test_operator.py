from __future__ import absolute_import

from collections import OrderedDict

from conftest import EVAL, dims, time, x, y, z, skipif_yask

import numpy as np
import pytest

from devito import (clear_cache, Grid, Eq, Operator, Constant, Function, Backward,
                    Forward, TimeFunction, SparseFunction, Dimension, configuration)
from devito.foreign import Operator as OperatorForeign
from devito.ir.iet import (Expression, Iteration, FindNodes, IsPerfectIteration,
                           retrieve_iteration_tree)


def dimify(dimensions):
    assert isinstance(dimensions, str)
    return tuple(dims()[i] for i in dimensions.split())


def symbol(name, dimensions, value=0., shape=(3, 5), mode='function'):
    """Short-cut for symbol creation to test "function"
    and "indexed" API."""
    assert(mode in ['function', 'indexed'])
    s = Function(name=name, dimensions=dimensions, shape=shape)
    s.data[:] = value
    return s.indexify() if mode == 'indexed' else s


@skipif_yask
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
        assert len(op.parameters) == 6
        assert op.parameters[0].name == 'a_dense'
        assert op.parameters[0].is_TensorArgument
        assert op.parameters[1].name == 'constant'
        assert op.parameters[1].is_ScalarArgument
        assert op.parameters[2].name == 'i_size'
        assert op.parameters[2].is_ScalarArgument
        assert op.parameters[3].name == 'i_s'
        assert op.parameters[3].is_ScalarArgument
        assert op.parameters[4].name == 'i_e'
        assert op.parameters[4].is_ScalarArgument
        assert op.parameters[5].name == 'timers'
        assert op.parameters[5].is_PtrArgument
        assert 'a_dense[i] = 2.0F*constant + a_dense[i]' in str(op.ccode)

    def test_logic_indexing(self):
        """
        Tests that logic indexing for stepping dimensions works.
        """
        grid = Grid(shape=(4, 4, 4))
        nt = 5

        u = Function(name='u', grid=grid)
        try:
            u.data[5]
            assert False
        except IndexError:
            pass

        v = TimeFunction(name='v', grid=grid, save=nt)
        try:
            v.data[nt]
            assert False
        except IndexError:
            pass

        v_mod = TimeFunction(name='v_mod', grid=grid)
        v_mod.data[0] = 1.
        v_mod.data[1] = 2.
        assert np.all(v_mod.data[0] == 1.)
        assert np.all(v_mod.data[1] == 2.)
        assert np.all(v_mod.data[2] == v_mod.data[0])
        assert np.all(v_mod.data[4] == v_mod.data[0])
        assert np.all(v_mod.data[3] == v_mod.data[1])
        assert np.all(v_mod.data[-1] == v_mod.data[1])
        assert np.all(v_mod.data[-2] == v_mod.data[0])


@skipif_yask
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
    @pytest.mark.parametrize('mode', ['function'])
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
        a = symbol(name='a', dimensions=(i, j, k, l), shape=(3, 5, 7, 6),
                   value=2., mode=mode)
        b = symbol(name='b', dimensions=(j, k), shape=(5, 7),
                   value=3., mode=mode)
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
        a = symbol(name='a', dimensions=(j, l), value=2., shape=(5, 6),
                   mode='indexed').base
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
        a = symbol(name='a', dimensions=(j, l), value=0., shape=(5, 6),
                   mode='indexed').base
        fa = a.function
        b = symbol(name='b', dimensions=(j, l), value=2., shape=(5, 6),
                   mode='indexed').base
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
        a = symbol(name='a', dimensions=(i, j, l), value=2., shape=(3, 5, 6),
                   mode='indexed').base
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
        i, j, l = dimify('i j l')
        a = Function(name='a', dimensions=(i, j, l), shape=(3, 5, 6)).indexed
        fa = a.function
        fa.data[0, :, :] = 2.

        eqn = eval(expr)
        Operator(eqn)(a=fa)
        assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)

    def test_constant_time_dense(self):
        """Test arithmetic between different data objects, namely Constant
        and Function."""
        i, j = dimify('i j')
        const = Constant(name='truc', value=2.)
        a = Function(name='a', shape=(20, 20), dimensions=(i, j))
        a.data[:] = 2.
        eqn = Eq(a, a + 2.*const)
        op = Operator(eqn)
        op.apply(a=a, truc=const)
        assert(np.allclose(a.data, 6.))

        # Applying a different constant still works
        op.apply(a=a, truc=Constant(name='truc2', value=3.))
        assert(np.allclose(a.data, 12.))


@skipif_yask
class TestAllocation(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    @pytest.mark.parametrize('shape', [(20, 20),
                                       (20, 20, 20),
                                       (20, 20, 20, 20)])
    def test_first_touch(self, shape):
        dimensions = dimify('i j k l')[:len(shape)]
        grid = Grid(shape=shape, dimensions=dimensions)
        m = Function(name='m', grid=grid, first_touch=True)
        assert(np.allclose(m.data, 0))
        m2 = Function(name='m2', grid=grid, first_touch=False)
        assert(np.allclose(m2.data, 0))
        assert(np.array_equal(m.data, m2.data))

    @pytest.mark.parametrize('staggered', [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
    ])
    def test_staggered(self, staggered):
        """
        Test the "deformed" allocation for staggered functions
        """
        grid = Grid(shape=tuple(11 for _ in staggered))
        f = Function(name='f', grid=grid, staggered=staggered)
        assert f.data.shape == tuple(11-i for i in staggered)
        # Add a non-staggered field to ensure that the auto-derived
        # dimension size arguments are at maximum
        g = Function(name='g', grid=grid)
        # Test insertion into a central point
        index = tuple(5 for _ in staggered)
        set_f = Eq(f.indexed[index], 2.)
        set_g = Eq(g.indexed[index], 3.)
        Operator([set_f, set_g])()
        assert f.data[index] == 2.

    @pytest.mark.parametrize('staggered', [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
        (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1),
    ])
    def test_staggered_time(self, staggered):
        """
        Test the "deformed" allocation for staggered functions
        """
        grid = Grid(shape=tuple(11 for _ in staggered[1:]))
        f = TimeFunction(name='f', grid=grid, staggered=staggered)
        assert f.data.shape[1:] == tuple(11-i for i in staggered[1:])
        # Add a non-staggered field to ensure that the auto-derived
        # dimension size arguments are at maximum
        g = TimeFunction(name='g', grid=grid)
        # Test insertion into a central point
        index = tuple([0] + [5 for _ in staggered[1:]])
        set_f = Eq(f.indexed[index], 2.)
        set_g = Eq(g.indexed[index], 3.)
        Operator([set_f, set_g])()
        assert f.data[index] == 2.


@skipif_yask
class TestArguments(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_override_cache_aliasing(self):
        """Test that the call-time overriding of Operator arguments works"""
        i, j, k, l = dimify('i j k l')
        shape = (3, 5, 7, 6)
        a = symbol(name='a', dimensions=(i, j, k, l), value=2.,
                   shape=shape, mode='indexed').base.function
        a1 = symbol(name='a', dimensions=(i, j, k, l), value=3.,
                    shape=shape, mode='indexed').base.function
        a2 = symbol(name='a', dimensions=(i, j, k, l), value=4.,
                    shape=shape, mode='indexed').base.function
        eqn = Eq(a, a+3)
        op = Operator(eqn)
        op()
        op(a=a1)
        op(a=a2)

        assert(np.allclose(a.data, np.zeros(shape) + 5))
        assert(np.allclose(a1.data, np.zeros(shape) + 6))
        assert(np.allclose(a2.data, np.zeros(shape) + 7))

    def test_override_symbol(self):
        """Test call-time symbols overrides with other symbols"""
        i, j, k, l = dimify('i j k l')
        shape = (3, 5, 7, 6)
        a = symbol(name='a', dimensions=(i, j, k, l), shape=shape, value=2.)
        a1 = symbol(name='a1', dimensions=(i, j, k, l), shape=shape, value=3.)
        a2 = symbol(name='a2', dimensions=(i, j, k, l), shape=shape, value=4.)
        op = Operator(Eq(a, a + 3))
        op()
        op(a=a1)
        op(a=a2)

        assert(np.allclose(a.data, np.zeros(shape) + 5))
        assert(np.allclose(a1.data, np.zeros(shape) + 6))
        assert(np.allclose(a2.data, np.zeros(shape) + 7))

    def test_override_array(self):
        """Test call-time symbols overrides with numpy arrays"""
        i, j, k, l = dimify('i j k l')
        shape = (3, 5, 7, 6)
        a = symbol(name='a', shape=shape, dimensions=(i, j, k, l), value=2.)
        a1 = np.zeros(shape=shape, dtype=np.float32) + 3.
        a2 = np.zeros(shape=shape, dtype=np.float32) + 4.
        op = Operator(Eq(a, a + 3))
        op()
        op(a=a1)
        op(a=a2)

        assert(np.allclose(a.data, np.zeros(shape) + 5))
        assert(np.allclose(a1, np.zeros(shape) + 6))
        assert(np.allclose(a2, np.zeros(shape) + 7))

    def test_dimension_size_infer(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        grid = Grid(shape=(3, 5, 7))
        a = Function(name='a', grid=grid)
        b = TimeFunction(name='b', grid=grid, save=nt)
        op = Operator(Eq(b, a))

        time = b.indices[0]
        op_arguments = op.arguments()
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt)

    def test_arg_offset_adjust(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        i, j, k = dimify('i j k')
        shape = (10, 10, 10)
        grid = Grid(shape=shape, dimensions=(i, j, k))
        a = Function(name='a', grid=grid).indexed
        b = TimeFunction(name='b', grid=grid, save=nt)
        time = b.indices[0]
        eqn = Eq(b.indexed[time + 1, i, j, k], b.indexed[time - 1, i, j, k]
                 + b.indexed[time, i, j, k] + a[i, j, k])
        op = Operator(eqn)
        args = {time.end_name: nt-10}
        op_arguments = op.arguments(**args)
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt - 8)

    def test_dimension_offset_adjust(self, nt=100):
        """Test that the dimension sizes are being inferred correctly"""
        i, j, k = dimify('i j k')
        shape = (10, 10, 10)
        grid = Grid(shape=shape, dimensions=(i, j, k))
        a = Function(name='a', grid=grid).indexed
        b = TimeFunction(name='b', grid=grid, save=nt)
        time = b.indices[0]
        eqn = Eq(b.indexed[time + 1, i, j, k], b.indexed[time - 1, i, j, k]
                 + b.indexed[time, i, j, k] + a[i, j, k])
        op = Operator(eqn)
        op_arguments = op.arguments(time=nt-10)
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt - 10)

    def test_dimension_size_override(self):
        """Test explicit overrides for the leading time dimension"""
        grid = Grid(shape=(3, 5, 7))
        a = TimeFunction(name='a', grid=grid)
        one = Function(name='one', grid=grid)
        one.data[:] = 1.
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
        i, j = dimify('i j')
        grid = Grid(shape=(10, 10), dimensions=(i, j))
        original_coords = (1., 1.)
        new_coords = (2., 2.)
        p_dim = Dimension('p_src')
        u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)
        src1 = SparseFunction(name='src1', grid=grid, dimensions=[time, p_dim],
                              npoint=1, nt=10, coordinates=original_coords)
        src2 = SparseFunction(name='src1', grid=grid, dimensions=[time, p_dim],
                              npoint=1, nt=10, coordinates=new_coords)
        op = Operator(src1.inject(u, src1))

        # Move the source from the location where the setup put it so we can test
        # whether the override picks up the original coordinates or the changed ones

        # Operator.arguments() returns a tuple of (data, dimension_sizes)
        args = op.arguments(src1=src2)
        arg_name = src1.name + "_coords"
        assert(np.array_equal(args[arg_name], np.asarray((new_coords,))))

    def test_start_end_2d(self):
        """ In a 2D square domain, ask the operator to operate over a smaller square
        and then ensure that it only operated on the requested region
        """
        i, j = dimify('i j')
        shape = (10, 10)
        start = 3
        end = 6
        a = Function(name='a', shape=shape, dimensions=(i, j))
        eqn = Eq(a, a + 1)
        op = Operator(eqn)
        op(i_s=start, i_e=end, j_s=start, j_e=end)
        mask = np.ones(shape, np.bool)
        mask[start:end, start:end] = 0
        assert(np.allclose(a.data[start:end, start:end], 1))
        assert(np.allclose(a.data[mask], 0))

    def test_start_end_3d(self):
        """ In a 3D cubical domain, ask the operator to operate over a smaller cube
        and then ensure that it only operated on the requested region
        """
        i, j, k = dimify('i j k')
        shape = (10, 10, 10)
        start = 3
        end = 6
        a = Function(name='a', shape=shape, dimensions=(i, j, k))
        eqn = Eq(a, a + 1)
        op = Operator(eqn)
        op(i_s=start, i_e=end, j_s=start, j_e=end, k_s=start, k_e=end)
        mask = np.ones(shape, np.bool)
        mask[start:end, start:end, start:end] = 0
        assert(np.allclose(a.data[start:end, start:end, start:end], 1))
        assert(np.allclose(a.data[mask], 0))

    def test_argument_derivation_order(self, nt=100):
        """ Ensure the precedence order of arguments is respected
        Defaults < (overriden by) Tensor Arguments < Dimensions < Scalar Arguments
        """
        i, j, k = dimify('i j k')
        shape = (10, 10, 10)
        grid = Grid(shape=shape, dimensions=(i, j, k))
        a = Function(name='a', grid=grid).indexed
        b_function = TimeFunction(name='b', grid=grid, save=nt)
        b = b_function.indexed
        time = b_function.indices[0]
        b1 = TimeFunction(name='b1', grid=grid, save=nt+1).indexed
        eqn = Eq(b[time, i, j, k], a[i, j, k])
        op = Operator(eqn)

        # Simple case, same as that tested above.
        # Repeated here for clarity of further tests.
        op_arguments = op.arguments()
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt)

        # Providing a tensor argument should infer the dimension size from its shape
        op_arguments = op.arguments(b=b1)
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt + 1)

        # Providing a dimension size explicitly should override the automatically inferred
        op_arguments = op.arguments(b=b1, time=nt - 1)
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt - 1)

        # Providing a scalar argument explicitly should override the automatically\
        # inferred
        op_arguments = op.arguments(b=b1, time=nt - 1, time_e=nt - 2)
        assert(op_arguments[time.start_name] == 0)
        assert(op_arguments[time.end_name] == nt - 2)


@skipif_yask
class TestDeclarator(object):

    @classmethod
    def setup_class(cls):
        clear_cache()

    def test_heap_1D_stencil(self, a, b):
        operator = Operator(Eq(a, a + b + 5.), dse='noop', dle=None)
        assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[i_size]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = i_s; i < i_e; i += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
  }
  gettimeofday(&end_section_0, NULL);
  timers->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)

    def test_heap_perfect_2D_stencil(self, a, c):
        operator = Operator([Eq(a, c), Eq(c, c*a)], dse='noop', dle=None)
        assert """\
  float (*c)[j_size];
  posix_memalign((void**)&c, 64, sizeof(float[i_size][j_size]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = i_s; i < i_e; i += 1)
  {
    for (int j = j_s; j < j_e; j += 1)
    {
      float s0 = c[i][j];
      c[i][j] = s0*c[i][j];
    }
  }
  gettimeofday(&end_section_0, NULL);
  timers->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(c);
  return 0;""" in str(operator.ccode)

    def test_heap_imperfect_2D_stencil(self, a, c):
        operator = Operator([Eq(a, 0.), Eq(c, c*a)], dse='noop', dle=None)
        assert """\
  float (*a);
  float (*c)[j_size];
  posix_memalign((void**)&a, 64, sizeof(float[i_size]));
  posix_memalign((void**)&c, 64, sizeof(float[i_size][j_size]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = i_s; i < i_e; i += 1)
  {
    a[i] = 0.0F;
    for (int j = j_s; j < j_e; j += 1)
    {
      c[i][j] = a[i]*c[i][j];
    }
  }
  gettimeofday(&end_section_0, NULL);
  timers->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(a);
  free(c);
  return 0;""" in str(operator.ccode)

    def test_stack_scalar_temporaries(self, a, t0, t1):
        operator = Operator([Eq(t0, 1.), Eq(t1, 2.), Eq(a, t0*t1*3.)],
                            dse='noop', dle=None)
        assert """\
  float (*a);
  posix_memalign((void**)&a, 64, sizeof(float[i_size]));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int i = i_s; i < i_e; i += 1)
  {
    float t0 = 1.00000000000000F;
    float t1 = 2.00000000000000F;
    a[i] = 3.0F*t0*t1;
  }
  gettimeofday(&end_section_0, NULL);
  timers->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  free(a);
  return 0;""" in str(operator.ccode)

    def test_stack_vector_temporaries(self, c_stack, e):
        operator = Operator([Eq(c_stack, e*1.)], dse='noop', dle=None)
        assert """\
  float c_stack[i_size][j_size] __attribute__((aligned(64)));
  struct timeval start_section_0, end_section_0;
  gettimeofday(&start_section_0, NULL);
  for (int k = k_s; k < k_e; k += 1)
  {
    for (int s = s_s; s < s_e; s += 1)
    {
      for (int q = q_s; q < q_e; q += 1)
      {
        for (int i = i_s; i < i_e; i += 1)
        {
          for (int j = j_s; j < j_e; j += 1)
          {
            c_stack[i][j] = 1.0F*e[k][s][q][i][j];
          }
        }
      }
    }
  }
  gettimeofday(&end_section_0, NULL);
  timers->section_0 += (double)(end_section_0.tv_sec-start_section_0.tv_sec)\
+(double)(end_section_0.tv_usec-start_section_0.tv_usec)/1000000;
  return 0;""" in str(operator.ccode)


@skipif_yask
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
            exprs = FindNodes(Expression).visit(tree[-1])
            assert len(exprs) == 3

    @pytest.mark.parametrize('exprs', [
        ('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])', 'Eq(ti1[x,y,z], ti3[x,y,z])',
         'Eq(ti3[x,y,z], ti1[x,y,z] + 1.)'),
        ('Eq(ti0[x,y,z], ti0[x,y,z-1] + ti1[x,y,z-1])', 'Eq(ti1[x,y,z], ti3[x,y,z-1])',
         'Eq(ti3[x,y,z], ti3[x,y,z-1] + ti0[x,y,z])'),
        ('Eq(ti0[x,y,z+2], ti0[x,y,z-1] + ti1[x,y,z+1])',
         'Eq(ti1[x,y,z+3], ti3[x,y,z+1])',
         'Eq(ti3[x,y,z+2], ti0[x,y,z+1]*ti3[x,y,z-1])'),
        ('Eq(ti0[x,y,z], ti0[x-2,y-1,z-1] + ti1[x+2,y+3,z+1])',
         'Eq(ti1[x+4,y+5,z+3], ti3[x+1,y-4,z+1])',
         'Eq(ti3[x+7,y,z+2], ti3[x+5,y,z-1] - ti0[x-3,y-2,z-4])')
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
            exprs = FindNodes(Expression).visit(tree[-1])
            assert len(exprs) == 3

    @pytest.mark.parametrize('exprs,axis,expected,visit', [
        # WAR 2->3; expected=2
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti3[x,y,z])',
          'Eq(ti3[x,y,z], ti1[x,y,z+1] + 1.)'),
         Forward, ['xyz'], 'xyz'),
        # WAR 1->2, 2->3; one may think it should be expected=3, but these are all
        # Arrays, so ti0 gets optimized through index bumping and array contraction,
        # which results in expected=2
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti0[x,y,z+1])',
          'Eq(ti3[x,y,z], ti1[x,y,z+2] + 1.)'),
         Forward, ['xyz', 'xyz'], 'xyzz'),
        # WAR 1->3; expected=1
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], ti3[x,y,z])',
          'Eq(ti3[x,y,z], ti0[x,y,z+1] + 1.)'),
         Forward, ['xyz'], 'xyz'),
        # WAR 1->2, 2->3; WAW 1->3; expected=2
        # ti0 is an Array, so the observation made above still holds (expected=2
        # rather than 3)
        (('Eq(ti0[x,y,z], ti0[x,y,z] + ti1[x,y,z])',
          'Eq(ti1[x,y,z], 3*ti0[x,y,z+2])',
          'Eq(ti0[x,y,0], ti0[x,y,0] + 1.)'),
         Forward, ['xyz', 'xy'], 'xyz'),
        # WAR 1->2; WAW 1->3; expected=3
        # Now tu, tv, tw are not Arrays, so they must end up in separate loops
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tu[t,x,y,0], tu[t,x,y,0] + 1.)'),
         Forward, ['txyz', 'txyz', 'txy'], 'txyzz'),
        # WAR 1->2; WAW 2->3; expected=3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tw[t,x,y,z], tv[t,x,y,z+3] + 1.)'),
         Forward, ['txyz', 'txyz', 'txyz'], 'txyzzz'),
        # WAR 1->2; WAW 1->3; expected=3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x+2,y,z])',
          'Eq(tu[t,3,y,0], tu[t,3,y,0] + 1.)'),
         Forward, ['txyz', 'txyz', 'ty'], 'txyzxyzy'),
        # WAR 1->2, WAW 2->3; expected=3
        (('Eq(tu[t,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tw[t,x,y,z], tv[t,x,y+1,z] + 1.)'),
         Forward, ['txyz', 'txyz', 'txyz'], 'txyzzyz'),
        # WAR 1->2; WAW 1->3; expected=3
        # Time Forward, anti dependence in time, end up in different loop nests
        (('Eq(tu[t-1,x,y,z], tu[t,x,y,z] + tv[t,x,y,z])',
          'Eq(tv[t,x,y,z], tu[t,x,y,z+2])',
          'Eq(tu[t-2,x,y,0], tu[t,x,y,0] + 1.)'),
         Forward, ['txyz', 'txyz', 'txy'], 'txyztxyztxy'),
        # Time Backward, so flow dependences in time
        (('Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
          'Eq(tv[t-1,x,y,z], tu[t,x,y,z+2])',
          'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + tv[t,x,y-1,z])'),
         Backward, ['txyz'], 'txyz'),
        # Time Backward, so flow dependences in time, interleaved with independent Eq
        (('Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
          'Eq(ti0[x,y,z], ti1[x,y,z+2])',
          'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + tv[t,x,y-1,z])'),
         Backward, ['txyz', 'xyz'], 'txyzxyz'),
        # Time Backward, so flow dependences in time, interleaved with dependent Eq
        (('Eq(ti0[x,y,z], ti1[x,y,z+2])',
          'Eq(tu[t-1,x,y,z], tu[t,x+3,y,z] + tv[t,x,y,z])',
          'Eq(tw[t-1,x,y,z], tu[t,x,y+1,z] + ti0[x,y-1,z])'),
         Backward, ['xyz', 'txyz'], 'xyztxyz')
    ])
    def test_consistency_anti_dependences(self, exprs, axis, expected, visit,
                                          ti0, ti1, ti3, tu, tv, tw):
        """
        Test that anti dependences end up generating multi loop nests, rather
        than a single loop nest enclosing all of the equations.
        """
        eq1, eq2, eq3 = EVAL(exprs, ti0.base, ti1.base, ti3.base,
                             tu.base, tv.base, tw.base)
        op = Operator([eq1, eq2, eq3], dse='noop', dle='noop', time_axis=axis)
        trees = retrieve_iteration_tree(op)
        assert len(trees) == len(expected)
        assert ["".join(i.dim.name for i in j) for j in trees] == expected
        iters = FindNodes(Iteration).visit(op)
        assert "".join(i.dim.name for i in iters) == visit

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
        grid = Grid(shape=(3, 3, 3), dimensions=(x, y, z), time_dimension=time)
        a = Function(name='a', grid=grid).indexed
        b = TimeFunction(name='b', grid=grid, save=6).indexed
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
        Test that equations using a mixture of Function and TimeFunction objects
        are embedded within the same time loop.
        """
        grid = Grid(shape=shape, dimensions=dimensions, time_dimension=time)
        a = TimeFunction(name='a', grid=grid, time_order=2, space_order=2)
        p_aux = Dimension(name='p_aux', size=10)
        b = Function(name='b', shape=shape + (10,), dimensions=dimensions + (p_aux,),
                     space_order=2)
        b.data[:] = 1.0
        b2 = Function(name='b2', shape=(10,) + shape, dimensions=(p_aux,) + dimensions,
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

        # Verify both operators produce the same result
        op(time=10)
        a.data[:] = 0.
        op2(time=10)

        for i in range(10):
            assert(np.allclose(b2.data[i, ...].reshape(-1) -
                               b.data[..., i].reshape(-1), 0.))

    def test_equations_mixed_timedim_stepdim(self):
        """"
        Test that two equations one using a TimeDimension the other a derived
        SteppingDimension end up in the same loop nest.
        """
        grid = Grid(shape=(3, 3, 3))
        x, y, z = grid.dimensions
        time = grid.time_dim
        t = grid.stepping_dim
        u1 = TimeFunction(name='u1', grid=grid)
        u2 = TimeFunction(name='u2', grid=grid, save=2)
        eqn_1 = Eq(u1.indexed[t+1, x, y, z], u1.indexed[t, x, y, z] + 1.)
        eqn_2 = Eq(u2.indexed[time+1, x, y, z], u2.indexed[time, x, y, z] + 1.)
        op = Operator([eqn_1, eqn_2], dse='noop', dle='noop')
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1
        assert len(trees[0][-1].nodes) == 2
        assert trees[0][-1].nodes[0].write == u1
        assert trees[0][-1].nodes[1].write == u2


@skipif_yask
@pytest.mark.skipif(configuration['backend'] != 'foreign',
                    reason="'foreign' wasn't selected as backend on startup")
class TestForeign(object):

    def test_explicit_run(self):
        time_dim = 6
        grid = Grid(shape=(11, 11))
        a = TimeFunction(name='a', grid=grid, time_order=1, save=time_dim)
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
