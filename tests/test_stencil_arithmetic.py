import numpy as np
import pytest
from sympy import Eq  # noqa

from devito import DenseData, Dimension, StencilKernel


@pytest.fixture
def x(xdim=4):
    return Dimension(name='x', size=xdim)


@pytest.fixture
def y(ydim=6):
    return Dimension(name='y', size=ydim)


@pytest.fixture
def ai(x, y, name='a', value=2.):
    a = DenseData(name=name, dimensions=(x, y))
    a.data[:] = value
    return a.indexify()


@pytest.fixture
def bi(x, y, name='b', value=3.):
    b = DenseData(name=name, dimensions=(x, y))
    b.data[:] = value
    return b.indexify()


@pytest.mark.parametrize('expr, result', [
    ('Eq(ai, ai + bi + 5.)', 10.),
    ('Eq(ai, bi - ai)', 1.),
    ('Eq(ai, 4 * (bi * ai))', 24.),
    ('Eq(ai, (6. / bi) + (8. * ai))', 18.),
])
def test_arithmetic_flat(ai, bi, expr, result):
    """Tests basic point-wise arithmetic on two-dimensional data"""
    eqn = eval(expr)
    StencilKernel(eqn)(ai.base.function, bi.base.function)
    assert np.allclose(ai.base.function.data, result, rtol=1e-12)


@pytest.mark.parametrize('expr, result', [
    ('Eq(ai, ai + bi + 5.)', 10.),
    ('Eq(ai, bi - ai)', 1.),
    ('Eq(ai, 4 * (bi * ai))', 24.),
    ('Eq(ai, (6. / bi) + (8. * ai))', 18.),
])
def test_arithmetic_deep(expr, result):
    """Tests basic point-wise arithmetic on multi-dimensional data"""
    i = Dimension(name='i', size=3)
    j = Dimension(name='j', size=4)
    k = Dimension(name='k', size=5)
    l = Dimension(name='l', size=6)

    a = DenseData(name='a', dimensions=(i, j, k, l))
    a.data[:] = 2.
    ai = a.indexify()
    b = DenseData(name='b', dimensions=(j, k))
    b.data[:] = 3.
    bi = b.indexify()

    eqn = eval(expr)
    StencilKernel(eqn)(ai.base.function, bi.base.function)
    assert np.allclose(ai.base.function.data, result, rtol=1e-12)
