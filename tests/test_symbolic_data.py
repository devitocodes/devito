import numpy as np
import pytest
from sympy import Derivative, as_finite_diff, simplify
from sympy.abc import h

from devito import DenseData, TimeData, clear_cache, t, x, y, z
from devito.interfaces import _SymbolCache


@pytest.fixture
def shape(xdim=20, ydim=30):
    return (xdim, ydim)


@pytest.mark.parametrize('SymbolType, dimension', [
    (DenseData, x), (DenseData, y),
    (TimeData, x), (TimeData, y), (TimeData, t),
])
def test_stencil_derivative(shape, SymbolType, dimension):
    """Test symbolic behaviour when expanding stencil derivatives"""
    u = SymbolType(name='u', shape=shape)
    u.data[:] = 66.6
    dx = u.diff(x)
    dxx = u.diff(x, x)
    # Check for sympy Derivative objects
    assert(isinstance(dx, Derivative) and isinstance(dxx, Derivative))
    s_dx = as_finite_diff(dx, [x - h, x])
    s_dxx = as_finite_diff(dxx, [x - h, x, x + h])
    # Check stencil length of first and second derivatives
    assert(len(s_dx.args) == 2 and len(s_dxx.args) == 3)
    u_dx = s_dx.args[0].args[1]
    u_dxx = s_dx.args[0].args[1]
    # Ensure that devito meta-data survived symbolic transformation
    assert(u_dx.shape[-2:] == shape and u_dxx.shape[-2:] == shape)
    assert(np.allclose(u_dx.data, 66.6))
    assert(np.allclose(u_dxx.data, 66.6))


@pytest.mark.parametrize('SymbolType, derivative, dim', [
    (DenseData, 'dx2', 3), (DenseData, 'dy2', 3),
    (TimeData, 'dx2', 3), (TimeData, 'dy2', 3), (TimeData, 'dt', 2)
])
def test_preformed_derivatives(shape, SymbolType, derivative, dim):
    """Test the stencil expressions provided by devito objects"""
    u = SymbolType(name='u', shape=shape, time_order=2, space_order=2)
    expr = getattr(u, derivative)
    assert(len(expr.args) == dim)


@pytest.mark.parametrize('derivative, dimension', [
    ('dx', x), ('dy', y), ('dz', z)
])
@pytest.mark.parametrize('order', [2, 4, 6, 8, 10, 12, 14, 16])
def test_derivatives_space(derivative, dimension, order):
    """Test second derivative expressions against native sympy"""
    u = TimeData(name='u', shape=(20, 20, 20), time_order=2, space_order=order)
    expr = getattr(u, derivative)
    # Establish native sympy derivative expression
    width = int(order / 2)
    indices = [(dimension + i * h) for i in range(-width, width + 1)]
    s_expr = as_finite_diff(u.diff(dimension), indices)
    assert(simplify(expr - s_expr) == 0)  # Symbolic equality
    assert(expr == s_expr)  # Exact equailty


@pytest.mark.parametrize('derivative, dimension', [
    ('dx2', x), ('dy2', y), ('dz2', z)
])
@pytest.mark.parametrize('order', [2, 4, 6, 8, 10, 12, 14, 16])
def test_second_derivatives_space(derivative, dimension, order):
    """Test second derivative expressions against native sympy"""
    u = TimeData(name='u', shape=(20, 20, 20), time_order=2, space_order=order)
    expr = getattr(u, derivative)
    # Establish native sympy derivative expression
    width = int(order / 2)
    indices = [(dimension + i * h) for i in range(-width, width + 1)]
    s_expr = as_finite_diff(u.diff(dimension, dimension), indices)
    assert(simplify(expr - s_expr) == 0)  # Symbolic equality
    assert(expr == s_expr)  # Exact equailty


def test_clear_cache(nx=1000, ny=1000):
    clear_cache()
    cache_size = len(_SymbolCache)

    for i in range(10):
        assert(len(_SymbolCache) == cache_size)

        DenseData(name='u', shape=(nx, ny), dtype=np.float64, space_order=2)

        assert(len(_SymbolCache) == cache_size + 1)

        clear_cache()
