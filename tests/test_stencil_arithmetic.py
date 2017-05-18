import numpy as np
import pytest
from sympy import Eq  # noqa

from devito import Operator, DenseData, TimeData, Dimension, time, t, x, y, z


@pytest.fixture
def i(size=3):
    return Dimension(name='i', size=size)


@pytest.fixture
def j(size=4):
    return Dimension(name='j', size=size)


@pytest.fixture
def k(size=5):
    return Dimension(name='k', size=size)


@pytest.fixture
def l(size=6):
    return Dimension(name='l', size=size)


def symbol(name, dimensions, value=0., mode='function'):
    """Short-cut for symbol creation to test "function"
    and "indexed" API."""
    assert(mode in ['function', 'indexed'])
    s = DenseData(name=name, dimensions=dimensions)
    s.data[:] = value
    return s.indexify() if mode == 'indexed' else s


@pytest.mark.parametrize('expr, result', [
    ('Eq(a, a + b + 5.)', 10.),
    ('Eq(a, b - a)', 1.),
    ('Eq(a, 4 * (b * a))', 24.),
    ('Eq(a, (6. / b) + (8. * a))', 18.),
])
@pytest.mark.parametrize('mode', ['function', 'indexed'])
def test_arithmetic_flat(i, j, expr, result, mode):
    """Tests basic point-wise arithmetic on two-dimensional data"""
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
def test_arithmetic_deep(i, j, k, l, expr, result, mode):
    """Tests basic point-wise arithmetic on multi-dimensional data"""
    a = symbol(name='a', dimensions=(i, j, k, l), value=2., mode=mode)
    b = symbol(name='b', dimensions=(j, k), value=3., mode=mode)
    fa = a.base.function if mode == 'indexed' else a
    fb = b.base.function if mode == 'indexed' else b

    eqn = eval(expr)
    Operator(eqn)(fa, fb)
    assert np.allclose(fa.data, result, rtol=1e-12)


@pytest.mark.parametrize('expr, result', [
    ('Eq(a[k, l], a[k - 1 , l] + 1.)',
     np.meshgrid(np.arange(2., 8.), np.arange(2., 7.))[1]),
    ('Eq(a[k, l], a[k, l - 1] + 1.)',
     np.meshgrid(np.arange(2., 8.), np.arange(2., 7.))[0]),
])
def test_arithmetic_indexed_increment(i, j, k, l, expr, result):
    """Tests point-wise increments with stencil offsets in one dimension"""
    a = symbol(name='a', dimensions=(k, l), value=2., mode='indexed').base
    fa = a.function
    fa.data[1:, 1:] = 0

    eqn = eval(expr)
    Operator(eqn)(fa)
    assert np.allclose(fa.data, result, rtol=1e-12)


@pytest.mark.parametrize('expr, result', [
    ('Eq(a[k, l], b[k - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
    ('Eq(a[k, l], b[k , l - 1] + 1.)', np.zeros((5, 6)) + 3.),
    ('Eq(a[k, l], b[k - 1, l - 1] + 1.)', np.zeros((5, 6)) + 3.),
    ('Eq(a[k, l], b[k + 1, l + 1] + 1.)', np.zeros((5, 6)) + 3.),
])
def test_arithmetic_indexed_stencil(i, j, k, l, expr, result):
    """Test point-wise arithmetic with stencil offsets across two
    functions in indexed expression format"""
    a = symbol(name='a', dimensions=(k, l), value=0., mode='indexed').base
    fa = a.function
    b = symbol(name='b', dimensions=(k, l), value=2., mode='indexed').base
    fb = b.function

    eqn = eval(expr)
    Operator(eqn)(fa, fb)
    assert np.allclose(fa.data[1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)


@pytest.mark.parametrize('expr, result', [
    ('Eq(a[1, k, l], a[0, k - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
    ('Eq(a[1, k, l], a[0, k , l - 1] + 1.)', np.zeros((5, 6)) + 3.),
    ('Eq(a[1, k, l], a[0, k - 1, l - 1] + 1.)', np.zeros((5, 6)) + 3.),
    ('Eq(a[1, k, l], a[0, k + 1, l + 1] + 1.)', np.zeros((5, 6)) + 3.),
])
def test_arithmetic_indexed_buffered(i, j, k, l, expr, result):
    """Test point-wise arithmetic with stencil offsets across a single
    functions with buffering dimension in indexed expression format"""
    a = symbol(name='a', dimensions=(i, k, l), value=2., mode='indexed').base
    fa = a.function

    eqn = eval(expr)
    Operator(eqn)(fa)
    assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)


@pytest.mark.parametrize('expr, result', [
    ('Eq(a[1, k, l], a[0, k - 1 , l] + 1.)', np.zeros((5, 6)) + 3.),
])
def test_arithmetic_indexed_open_loops(i, j, k, l, expr, result):
    """Test point-wise arithmetic with stencil offsets and open loop
    boundaries in indexed expression format"""
    k.size = None
    l.size = None
    a = DenseData(name='a', dimensions=(i, k, l), shape=(3, 5, 6)).indexed
    fa = a.function
    fa.data[0, :, :] = 2.

    eqn = eval(expr)
    Operator(eqn)(fa)
    assert np.allclose(fa.data[1, 1:-1, 1:-1], result[1:-1, 1:-1], rtol=1e-12)


def test_override(i, j, k, l):
    """Test that the call-time overriding of Operator arguments works"""
    a = symbol(name='a', dimensions=(i, j, k, l), value=2., mode='indexed').base.function
    a1 = symbol(name='a', dimensions=(i, j, k, l), value=3., mode='indexed').base.function
    a2 = symbol(name='a', dimensions=(i, j, k, l), value=4., mode='indexed').base.function
    eqn = Eq(a, a+3)
    op = Operator(eqn)
    op()
    op(a=a1)
    op(a=a2)
    shape = [d.size for d in [i, j, k, l]]

    assert(np.allclose(a.data, np.zeros(shape) + 5))
    assert(np.allclose(a1.data, np.zeros(shape) + 6))
    assert(np.allclose(a2.data, np.zeros(shape) + 7))


def test_dimension_size_infer(i, j, k, nt=100):
    """Test that the dimension sizes are being inferred correctly"""
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
