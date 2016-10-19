import numpy as np
import pytest
from sympy import Eq  # noqa

from devito import DenseData, Dimension, StencilKernel


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


@pytest.mark.parametrize('expr, result', [
    ('Eq(ai, ai + bi + 5.)', 10.),
    ('Eq(ai, bi - ai)', 1.),
    ('Eq(ai, 4 * (bi * ai))', 24.),
    ('Eq(ai, (6. / bi) + (8. * ai))', 18.),
])
def test_arithmetic_flat(i, j, expr, result):
    """Tests basic point-wise arithmetic on two-dimensional data"""

    a = DenseData(name='a', dimensions=(i, j))
    a.data[:] = 2.
    ai = a.indexify()
    b = DenseData(name='b', dimensions=(i, j))
    b.data[:] = 3.
    bi = b.indexify()

    eqn = eval(expr)
    StencilKernel(eqn)(ai.base.function, bi.base.function)
    assert np.allclose(ai.base.function.data, result, rtol=1e-12)


@pytest.mark.parametrize('expr, result', [
    ('Eq(ai, ai + bi + 5.)', 10.),
    ('Eq(ai, bi - ai)', 1.),
    ('Eq(ai, 4 * (bi * ai))', 24.),
    ('Eq(ai, (6. / bi) + (8. * ai))', 18.),
])
def test_arithmetic_deep(i, j, k, l, expr, result):
    """Tests basic point-wise arithmetic on multi-dimensional data"""

    a = DenseData(name='a', dimensions=(i, j, k, l))
    a.data[:] = 2.
    ai = a.indexify()
    b = DenseData(name='b', dimensions=(j, k))
    b.data[:] = 3.
    bi = b.indexify()

    eqn = eval(expr)
    StencilKernel(eqn)(ai.base.function, bi.base.function)
    assert np.allclose(ai.base.function.data, result, rtol=1e-12)
