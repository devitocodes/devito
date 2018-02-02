import numpy as np
import pytest
from conftest import skipif_yask
from sympy import Derivative, simplify

from devito import Grid, Function, TimeFunction


@pytest.fixture
def shape(xdim=20, ydim=30, zdim=20):
    return (xdim, ydim, zdim)


@pytest.fixture
def grid(shape):
    return Grid(shape=shape)


@pytest.fixture
def x(grid):
    return grid.dimensions[0]


@pytest.fixture
def y(grid):
    return grid.dimensions[1]


@pytest.fixture
def z(grid):
    return grid.dimensions[2]


@pytest.fixture
def t(grid):
    return grid.stepping_dim


@skipif_yask
@pytest.mark.parametrize('SymbolType, dim', [
    (Function, x), (Function, y),
    (TimeFunction, x), (TimeFunction, y), (TimeFunction, t),
])
def test_stencil_derivative(grid, shape, SymbolType, dim):
    """Test symbolic behaviour when expanding stencil derivatives"""
    i = dim(grid)  # issue fixtures+parametrize: github.com/pytest-dev/pytest/issues/349
    u = SymbolType(name='u', grid=grid)
    u.data[:] = 66.6
    di = u.diff(i)
    dii = u.diff(i, i)
    # Check for sympy Derivative objects
    assert(isinstance(di, Derivative) and isinstance(dii, Derivative))
    s_di = di.as_finite_difference([i - i.spacing, i])
    s_dii = dii.as_finite_difference([i - i.spacing, i, i + i.spacing])
    # Check stencil length of first and second derivatives
    assert(len(s_di.args) == 2 and len(s_dii.args) == 3)
    u_di = s_di.args[0].args[1]
    u_dii = s_di.args[0].args[1]
    # Ensure that devito meta-data survived symbolic transformation
    assert(u_di.grid.shape == shape and u_dii.grid.shape == shape)
    assert(u_di.shape == u.shape and u_dii.shape == u.shape)
    assert(np.allclose(u_di.data, 66.6))
    assert(np.allclose(u_dii.data, 66.6))


@skipif_yask
@pytest.mark.parametrize('SymbolType, derivative, dim', [
    (Function, 'dx2', 3), (Function, 'dy2', 3),
    (TimeFunction, 'dx2', 3), (TimeFunction, 'dy2', 3), (TimeFunction, 'dt', 2)
])
def test_preformed_derivatives(grid, SymbolType, derivative, dim):
    """Test the stencil expressions provided by devito objects"""
    u = SymbolType(name='u', grid=grid, time_order=2, space_order=2)
    expr = getattr(u, derivative)
    assert(len(expr.args) == dim)


@skipif_yask
@pytest.mark.parametrize('derivative, dim', [
    ('dx', x), ('dy', y), ('dz', z)
])
@pytest.mark.parametrize('order', [1, 2, 4, 6, 8, 10, 12, 14, 16])
def test_derivatives_space(grid, derivative, dim, order):
    """Test first derivative expressions against native sympy"""
    dim = dim(grid)  # issue fixtures+parametrize: github.com/pytest-dev/pytest/issues/349
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=order)
    expr = getattr(u, derivative)
    # Establish native sympy derivative expression
    width = int(order / 2)
    if order == 1:
        indices = [dim, dim + dim.spacing]
    else:
        indices = [(dim + i * dim.spacing) for i in range(-width, width + 1)]
    s_expr = u.diff(dim).as_finite_difference(indices)
    assert(simplify(expr - s_expr) == 0)  # Symbolic equality
    assert(expr == s_expr)  # Exact equailty


@skipif_yask
@pytest.mark.parametrize('derivative, dim', [
    ('dx2', x), ('dy2', y), ('dz2', z)
])
@pytest.mark.parametrize('order', [2, 4, 6, 8, 10, 12, 14, 16])
def test_second_derivatives_space(grid, derivative, dim, order):
    """Test second derivative expressions against native sympy"""
    dim = dim(grid)  # issue fixtures+parametrize: github.com/pytest-dev/pytest/issues/349
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=order)
    expr = getattr(u, derivative)
    # Establish native sympy derivative expression
    width = int(order / 2)
    indices = [(dim + i * dim.spacing) for i in range(-width, width + 1)]
    s_expr = u.diff(dim, dim).as_finite_difference(indices)
    assert(simplify(expr - s_expr) == 0)  # Symbolic equality
    assert(expr == s_expr)  # Exact equailty
