import numpy as np
import pytest
from conftest import skipif_yask
from sympy import Derivative, simplify, diff

from devito import (Grid, Function, TimeFunction, Eq, Operator,
                    clear_cache, ConditionalDimension)


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


@skipif_yask
@pytest.mark.parametrize('space_order', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# Only test x and t as y and z are the same as x
@pytest.mark.parametrize('derivative', ['dx', 'dxl', 'dxr', 'dx2'])
def test_fd_space(derivative, space_order):
    """
    This test compare the discrete finite-difference scheme against polynomials
    For a given order p, the fiunite difference scheme should
    be exact for polynomials of order p
    :param derivative: name of the derivative to be tested
    :param space_order: space order of the finite difference stencil
    """
    clear_cache()
    # dummy axis dimension
    nx = 100
    xx = np.linspace(-1, 1, nx)
    dx = xx[1] - xx[0]
    # Symbolic data
    grid = Grid(shape=(nx,), dtype=np.float32)
    x = grid.dimensions[0]
    u = Function(name="u", grid=grid, space_order=space_order)
    du = Function(name="du", grid=grid, space_order=space_order)
    # Define polynomial with exact fd
    coeffs = np.ones((space_order,), dtype=np.float32)
    polynome = sum([coeffs[i]*x**i for i in range(0, space_order)])
    polyvalues = np.array([polynome.subs(x, xi) for xi in xx], np.float32)
    # Fill original data with the polynomial values
    u.data[:] = polyvalues
    # True derivative of the polynome
    Dpolynome = diff(diff(polynome)) if derivative == 'dx2' else diff(polynome)
    Dpolyvalues = np.array([Dpolynome.subs(x, xi) for xi in xx], np.float32)
    # FD derivative, symbolic
    u_deriv = getattr(u, derivative)
    # Compute numerical FD
    stencil = Eq(du, u_deriv)
    op = Operator(stencil, subs={x.spacing: dx})
    op.apply()

    # Check exactness of the numerical derivative except inside space_brd
    space_border = space_order
    error = abs(du.data[space_border:-space_border] -
                Dpolyvalues[space_border:-space_border])
    assert np.isclose(np.mean(error), 0., atol=1e-3)


@skipif_yask
def test_subsampled_fd():
    """
    Test that the symbolic interface is working for space subsampled
    functions.
    """
    nt = 19
    grid = Grid(shape=(12, 12), extent=(11, 11))

    u = TimeFunction(name='u', grid=grid, save=nt, space_order=2)
    assert(grid.time_dim in u.indices)

    # Creates subsampled spatial dimensions and according grid
    dims = tuple([ConditionalDimension(d.name+'sub', parent=d, factor=2)
                  for d in u.grid.dimensions])
    grid2 = Grid((6, 6), dimensions=dims)
    u2 = TimeFunction(name='u2', grid=grid2, save=nt, space_order=1)
    for i in range(nt):
        for j in range(u2.data_allocated.shape[2]):
            u2.data_allocated[i, :, j] = np.arange(u2.data_allocated.shape[2])

    eqns = [Eq(u.forward, u + 1.), Eq(u2.forward, u2.dx)]
    op = Operator(eqns, dse="advanced")
    op.apply(time_M=nt-2)
    # Verify that u2[1, x,y]= du2/dx[0, x, y]

    assert np.allclose(u.data[-1], nt-1)
    assert np.allclose(u2.data[1], 0.5)
