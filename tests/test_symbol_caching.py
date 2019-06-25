import weakref

import numpy as np
import pytest

from conftest import skipif
from devito import (Grid, Function, TimeFunction, SparseFunction, SparseTimeFunction,
                    ConditionalDimension, SubDimension, Constant, Operator, Eq, Dimension,
                    clear_cache)
from devito.types.basic import _SymbolCache, Scalar

pytestmark = skipif(['yask', 'ops'])


@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_cache_function_new(FunctionType):
    """Test that new u[x, y] instances don't cache"""
    grid = Grid(shape=(3, 4))
    u0 = FunctionType(name='u', grid=grid)
    u0.data[:] = 6.
    u1 = FunctionType(name='u', grid=grid)
    u1.data[:] = 2.
    assert np.allclose(u0.data, 6.)
    assert np.allclose(u1.data, 2.)


@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_cache_function_same_indices(FunctionType):
    """Test caching of derived u[x, y] instance from derivative"""
    grid = Grid(shape=(3, 4))
    u0 = FunctionType(name='u', grid=grid)
    u0.data[:] = 6.
    # Pick u(x, y) and u(x + h_x, y) from derivative
    u1 = u0.dx.evaluate.args[1].args[2]
    u2 = u0.dx.evaluate.args[0].args[1]
    assert np.allclose(u1.data, 6.)
    assert np.allclose(u2.data, 6.)


@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_cache_function_different_indices(FunctionType):
    """Test caching of u[x + h, y] instance from derivative"""
    grid = Grid(shape=(3, 4))
    u0 = FunctionType(name='u', grid=grid)
    u0.data[:] = 6.
    # Pick u[x + h, y] (different indices) from derivative
    u = u0.dx.evaluate.args[0].args[1]
    assert np.allclose(u.data, u0.data)


def test_cache_constant_new():
    """Test that new u[x, y] instances don't cache"""
    u0 = Constant(name='u')
    u0.data = 6.
    u1 = Constant(name='u')
    u1.data = 2.
    assert u0.data == 6.
    assert u1.data == 2.


def test_symbol_cache_aliasing():
    """Test to assert that our aliasing cache isn't defeated by sympys
    non-aliasing symbol cache.

    For further explanation consider the symbol u[x, y] and it's first
    derivative in x, which includes the symbols u[x, y] and u[x + h, y].
    The two functions are aliased in devito's caching mechanism to allow
    multiple stencil indices pointing at the same data object u, but
    SymPy treats these two instances as separate functions and thus is
    allowed to delete one or the other when the cache is cleared.

    The test below asserts that u[x + h, y] is deleted, the data on u
    is still intact through our own caching mechanism."""

    # Ensure a clean cache to start with
    clear_cache()
    # FIXME: Currently not working, presumably due to our
    # failure to cache new instances?
    # assert(len(_SymbolCache) == 0)

    # Create first instance of u and fill its data
    grid = Grid(shape=(3, 4))
    u = Function(name='u', grid=grid)
    u.data[:] = 6.
    u_ref = weakref.ref(u.data)

    # Create u[x + h, y] and delete it again
    dx = u.dx  # Contains two u symbols: u[x, y] and u[x + h, y]
    del dx
    clear_cache()
    # FIXME: Unreliable cache sizes
    # assert len(_SymbolCache) == 1  # We still have a reference to u
    assert np.allclose(u.data, 6.)  # u.data is alive and well

    # Remove the final instance and ensure u.data got deallocated
    del u
    clear_cache()
    assert u_ref() is None


def test_symbol_cache_aliasing_reverse():
    """Test to assert that removing he original u[x, y] instance does
    not impede our alisaing cache or leaks memory.
    """

    # Ensure a clean cache to start with
    clear_cache()
    # FIXME: Currently not working, presumably due to our
    # failure to cache new instances?
    # assert(len(_SymbolCache) == 0)

    # Create first instance of u and fill its data
    grid = Grid(shape=(3, 4))
    u = Function(name='u', grid=grid)
    u.data[:] = 6.
    u_ref = weakref.ref(u.data)

    # Create derivative and delete orignal u[x, y]
    dx = u.dx
    del u
    clear_cache()
    # We still have a references to u
    # FIXME: Unreliable cache sizes
    # assert len(_SymbolCache) == 1
    # Ensure u[x + h, y] still holds valid data
    assert np.allclose(dx.evaluate.args[0].args[1].data, 6.)

    del dx
    clear_cache()
    # FIXME: Unreliable cache sizes
    # assert len(_SymbolCache) == 0  # We still have a reference to u_h
    assert u_ref() is None


def test_clear_cache(nx=1000, ny=1000):
    grid = Grid(shape=(nx, ny), dtype=np.float64)
    clear_cache()
    cache_size = len(_SymbolCache)

    for i in range(10):
        assert(len(_SymbolCache) == cache_size)

        Function(name='u', grid=grid, space_order=2)

        assert(len(_SymbolCache) == cache_size + 1)

        clear_cache()


def test_cache_after_indexification():
    """Test to assert that the SymPy cache retrieves the right Devito data object
    after indexification.
    """
    grid = Grid(shape=(4, 4, 4))
    u0 = Function(name='u', grid=grid, space_order=0)
    u1 = Function(name='u', grid=grid, space_order=1)
    u2 = Function(name='u', grid=grid, space_order=2)

    for i in [u0, u1, u2]:
        assert i.indexify().base.function.space_order ==\
            (i.indexify() + 1.).args[1].base.function.space_order


def test_constant_hash():
    """Test that different Constants have different hash value."""
    c0 = Constant(name='c')
    c1 = Constant(name='c')
    assert c0 is not c1
    assert hash(c0) != hash(c1)


@pytest.mark.parametrize('FunctionType', [Function, TimeFunction])
def test_function_hash(FunctionType):
    """Test that different Functions have different hash value."""
    grid0 = Grid(shape=(3, 3))
    u0 = FunctionType(name='u', grid=grid0)
    grid1 = Grid(shape=(4, 4))
    u1 = FunctionType(name='u', grid=grid1)
    assert u0 is not u1
    assert hash(u0) != hash(u1)
    # Now with the same grid
    u2 = FunctionType(name='u', grid=grid0)
    assert u0 is not u2
    assert hash(u0) != hash(u2)


@pytest.mark.parametrize('FunctionType', [SparseFunction, SparseTimeFunction])
def test_sparse_function_hash(FunctionType):
    """Test that different Functions have different hash value."""
    grid0 = Grid(shape=(3, 3))
    u0 = FunctionType(name='u', grid=grid0, npoint=1, nt=10)
    grid1 = Grid(shape=(4, 4))
    u1 = FunctionType(name='u', grid=grid1, npoint=1, nt=10)
    assert u0 is not u1
    assert hash(u0) != hash(u1)
    # Now with the same grid
    u2 = FunctionType(name='u', grid=grid0, npoint=1, nt=10)
    assert u0 is not u2
    assert hash(u0) != hash(u2)


def test_scalar_cache():
    """
    Test that Scalars with same name but different attributes do not alias to
    the same Scalar. Conversely, if the name and the attributes are the same,
    they must alias to the same Scalar.
    """
    s0 = Scalar(name='s0')
    s1 = Scalar(name='s0')
    assert s0 is s1

    s2 = Scalar(name='s0', dtype=np.int32)
    assert s2 is not s1

    s3 = Scalar(name='s0', is_const=True)
    assert s3 is not s1


def test_dimension_cache():
    """
    Test that Dimensions with same name but different attributes do not alias to
    the same Dimension. Conversely, if the name and the attributes are the same,
    they must alias to the same Dimension.
    """
    d0 = Dimension(name='d')
    d1 = Dimension(name='d')
    assert d0 is d1

    s0 = Scalar(name='s0')
    s1 = Scalar(name='s1')

    d2 = Dimension(name='d', spacing=s0)
    d3 = Dimension(name='d', spacing=s1)
    assert d2 is not d3

    d4 = Dimension(name='d', spacing=s1)
    assert d3 is d4

    d5 = Dimension(name='d', spacing=Constant(name='s1'))
    assert d2 is not d5


def test_conditional_dimension_cache():
    """
    Test that ConditionalDimensions with same name but different attributes do not
    alias to the same ConditionalDimension. Conversely, if the name and the attributes
    are the same, they must alias to the same ConditionalDimension.
    """
    i = Dimension(name='i')
    ci0 = ConditionalDimension(name='ci', parent=i, factor=4)
    ci1 = ConditionalDimension(name='ci', parent=i, factor=4)
    assert ci0 is ci1

    ci2 = ConditionalDimension(name='ci', parent=i, factor=8)
    assert ci2 is not ci1

    ci3 = ConditionalDimension(name='ci', parent=i, factor=4, indirect=True)
    assert ci3 is not ci1

    s = Scalar(name='s')
    ci4 = ConditionalDimension(name='ci', parent=i, factor=4, condition=s > 3)
    assert ci4 is not ci1
    ci5 = ConditionalDimension(name='ci', parent=i, factor=4, condition=s > 3)
    assert ci5 is ci4


def test_sub_dimension_cache():
    """
    Test that SubDimensions with same name but different attributes do not
    alias to the same SubDimension. Conversely, if the name and the attributes
    are the same, they must alias to the same SubDimension.
    """
    x = Dimension('x')
    xi0 = SubDimension.middle('xi', x, 1, 1)
    xi1 = SubDimension.middle('xi', x, 1, 1)
    assert xi0 is xi1

    xl0 = SubDimension.left('xl', x, 2)
    xl1 = SubDimension.left('xl', x, 2)
    assert xl0 is xl1
    xl2asxi = SubDimension.left('xi', x, 2)
    assert xl2asxi is not xl1
    assert xl2asxi is not xi1

    xr0 = SubDimension.right('xr', x, 1)
    xr1 = SubDimension.right('xr', x, 1)
    assert xr0 is xr1


def test_operator_leakage_function():
    """
    Test to ensure that Operator creation does not cause memory leaks for (Time)Functions.
    """
    grid = Grid(shape=(5, 6))
    f = Function(name='f', grid=grid)
    g = TimeFunction(name='g', grid=grid)

    # Take weakrefs to test whether symbols are dead or alive
    w_f = weakref.ref(f)
    w_g = weakref.ref(g)

    # Create operator and delete everything again
    op = Operator(Eq(f, 2 * g))
    w_op = weakref.ref(op)
    del op
    del f
    del g
    clear_cache()

    # Test whether things are still hanging around
    assert w_f() is None
    assert w_g() is None
    assert w_op() is None


def test_operator_leakage_sparse():
    """
    Test to ensure that Operator creation does not cause memory leaks for
    SparseTimeFunctions.
    """
    grid = Grid(shape=(5, 6))
    a = Function(name='a', grid=grid)
    s = SparseTimeFunction(name='s', grid=grid, npoint=1, nt=1)
    w_a = weakref.ref(a)
    w_s = weakref.ref(s)

    # Create operator and delete everything again
    op = Operator(s.interpolate(a))
    w_op = weakref.ref(op)
    del op
    del s
    del a
    clear_cache()

    # Test whether things are still hanging around
    assert w_a() is None
    assert w_s() is None
    assert w_op() is None
