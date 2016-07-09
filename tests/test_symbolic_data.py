from devito import DenseData, TimeData
from sympy import symbols, Derivative, as_finite_diff
import pytest
import numpy as np


@pytest.fixture
def shape(xdim=20, ydim=30):
    return (xdim, ydim)


@pytest.mark.parametrize('SymbolType, dimension', [
    (DenseData, 'x'), (DenseData, 'y'),
    (TimeData, 'x'), (TimeData, 'y'), (TimeData, 't'),
])
def test_stencil_derivative(shape, SymbolType, dimension):
    """Test symbolic behaviour when expanding stencil derivatives"""
    x, h = symbols('%s h' % dimension)
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
    u = SymbolType(name='u', shape=shape)
    expr = getattr(u, derivative)
    assert(len(expr.args) == dim)
